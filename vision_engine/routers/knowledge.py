"""Knowledge router for MVE — proxy to the monitait_knowledge (MKB) service +
the kb_* helper functions routers/ai.py uses for RAG.

Wired in vision_engine/main.py before commands_router (the catch-all):

    from routers.knowledge import router as knowledge_router
    app.include_router(knowledge_router)

Design contract (do not weaken):
  - Every call is timeout-bounded and returns a structured error instead of
    raising. MKB being down must DEGRADE the assistant, never break the
    dashboard or the capture loop.
  - kb_available() gates whether ai.py advertises the KB tools at all.

Hardened vs the original addon (2026-07-28):
  - kb_available() has a TTL so one MKB blip can't disable the KB for the whole
    process life (the old cache latched False forever).
  - knowledge_status() feeds the same liveness cache (one source of truth).
  - the proxy runs the blocking requests call off the event loop
    (asyncio.to_thread) so a large upload can't freeze every MVE endpoint.
  - timeouts are read with a safe-int helper (a bad env value can't crash the
    ai.py import at load time).
"""
import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)
router = APIRouter()

KB_URL = os.environ.get("KB_URL", "http://knowledge:4445")


def _safe_int(env_name: str, default: int) -> int:
    try:
        return int(os.environ.get(env_name, str(default)))
    except (TypeError, ValueError):
        logger.warning("MVE: bad %s env value, using default %s", env_name, default)
        return default


KB_TIMEOUT = _safe_int("KB_TIMEOUT", 30)
# Uploads and reindex runs are slow by nature; own budget rather than raising
# the timeout for every call.
KB_LONG_TIMEOUT = _safe_int("KB_LONG_TIMEOUT", 300)

# Liveness cache. `_available` is the last probe result; `_available_ts` is when
# we probed (monotonic). A TTL re-probe means a transient MKB outage disables
# the tools only until it recovers, not forever.
_available: Optional[bool] = None
_available_ts: float = 0.0
_AVAILABLE_TTL = 45.0
# Guards the single in-flight background probe so a burst of callers spawns at
# most one health request.
_probe_lock = threading.Lock()
_probing = False


def _mark(alive: bool) -> bool:
    global _available, _available_ts
    _available = alive
    _available_ts = time.monotonic()
    return alive


def _probe_kb_blocking() -> bool:
    try:
        resp = requests.get(f"{KB_URL}/health", timeout=3)
        return _mark(resp.status_code == 200)
    except Exception:
        return _mark(False)


def _probe_kb_background() -> None:
    """Refresh the liveness cache off the request path. Never raises."""
    global _probing
    try:
        _probe_kb_blocking()
    finally:
        with _probe_lock:
            _probing = False


def kb_available(force: bool = False) -> bool:
    """NON-BLOCKING cached liveness check. Returns the last known result
    INSTANTLY and refreshes in a background thread when the cache goes stale,
    so a KB outage can never hold a core request thread (charts, status, the
    capture loop). ai.py calls this on the AI-query path; making it block would
    couple that path — and, via the shared threadpool, everything else — to the
    health of a service that is allowed to be absent.

    First-ever call returns False (assume-down until proven up — the safe
    default that never waits) and kicks off the first probe. `force=True` does a
    synchronous probe; it is used ONLY by /api/knowledge/status, which the
    Knowledge tab hits deliberately and is never in a core code path."""
    global _probing
    if force:
        return _probe_kb_blocking()
    now = time.monotonic()
    if _available is not None and (now - _available_ts) < _AVAILABLE_TTL:
        return _available
    # Stale or first-ever: trigger at most one background probe, return the last
    # known value immediately (False if we have never successfully probed).
    with _probe_lock:
        if not _probing:
            _probing = True
            threading.Thread(target=_probe_kb_background, daemon=True).start()
    return _available if _available is not None else False


def _fail(exc: Exception, what: str) -> Dict[str, Any]:
    _mark(False)
    logger.warning("MVE: knowledge service unavailable during %s: %s", what, exc)
    return {"error": "knowledge_service_unavailable", "detail": str(exc), "hits": [],
            "citations": [], "found": False}


# ---------------------------------------------------------------------------
# helpers used by routers/ai.py
# ---------------------------------------------------------------------------
def kb_search(query: str, top_k: int = 8, **filters) -> Dict[str, Any]:
    try:
        resp = requests.post(f"{KB_URL}/api/kb/search",
                             json={"query": query, "top_k": top_k, **filters},
                             timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return _fail(exc, "search")


def kb_context(query: str, token_budget: int = 3000, **filters) -> Dict[str, Any]:
    """Citation-numbered passages, ready to drop into a prompt."""
    try:
        resp = requests.post(f"{KB_URL}/api/kb/context",
                             json={"query": query, "token_budget": token_budget, **filters},
                             timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return _fail(exc, "context")


def kb_similar_issues(text: str, limit: int = 5) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{KB_URL}/api/kb/similar_issues",
                            params={"text": text, "limit": limit}, timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return _fail(exc, "similar_issues")


def kb_list_db_sources() -> Dict[str, Any]:
    """Registered factory databases the assistant may query live (name/type/id)."""
    try:
        resp = requests.get(f"{KB_URL}/api/kb/db_sources", timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return _fail(exc, "list_db_sources")


def kb_describe_db(source_id: int) -> Dict[str, Any]:
    """Tables + columns of a factory DB — the text-to-SQL context."""
    try:
        resp = requests.get(f"{KB_URL}/api/kb/db_sources/{source_id}/schema", timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return _fail(exc, "describe_db")


def kb_query_db(source_id: int, sql: str, max_rows: int = 200) -> Dict[str, Any]:
    """Run ONE read-only SELECT against a factory DB (guarded service-side)."""
    try:
        resp = requests.post(f"{KB_URL}/api/kb/db_sources/{source_id}/query",
                             json={"sql": sql, "max_rows": max_rows}, timeout=KB_TIMEOUT)
        # 400 = the read-only guard or a SQL error; surface it to the model
        if resp.status_code == 400:
            return {"error": (resp.json() or {}).get("detail", "query rejected")}
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return _fail(exc, "query_db")


def kb_resolve_asset(camera_id: Optional[int] = None, procedure: Optional[str] = None,
                     channel: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Map an alarm's camera/procedure/channel to a plant asset so retrieval can
    be boosted to the machine that is actually complaining."""
    try:
        params = {k: v for k, v in {"camera_id": camera_id, "procedure": procedure,
                                    "channel": channel}.items() if v is not None}
        if not params:
            return None
        resp = requests.get(f"{KB_URL}/api/kb/assets/resolve", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("asset")
    except Exception:
        return None


def kb_federated_tools() -> List[Dict[str, Any]]:
    """Namespaced external MCP tools, ready to append to the assistant's tool
    array. Empty list on any failure — a broken ERP connector must not remove
    the assistant's own tools. (Federation is out of scope for the RAG MVP.)"""
    try:
        resp = requests.get(f"{KB_URL}/api/kb/mcp/tools", timeout=10)
        resp.raise_for_status()
        return resp.json().get("tools", [])
    except Exception as exc:
        logger.debug("MVE: federated tool list unavailable: %s", exc)
        return []


def kb_call_federated(namespaced_tool: str, arguments: Dict[str, Any],
                      issue_id: Optional[int] = None) -> Dict[str, Any]:
    """Invoke `<server>__<tool>`."""
    if "__" not in namespaced_tool:
        return {"error": f"not a federated tool name: {namespaced_tool}"}
    server, tool = namespaced_tool.split("__", 1)
    try:
        resp = requests.post(f"{KB_URL}/api/kb/mcp/call",
                             json={"server": server, "tool": tool,
                                   "arguments": arguments, "issue_id": issue_id},
                             timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": f"MCP call failed: {exc}", "server": server, "tool": tool}


def kb_propose_action(title: str, endpoint: str, body: Dict[str, Any],
                      rationale: str = "", issue_id: Optional[int] = None,
                      method: str = "POST") -> Dict[str, Any]:
    """Record an inert proposal. The assistant may never apply it itself."""
    try:
        resp = requests.post(f"{KB_URL}/api/kb/actions/propose",
                             json={"title": title, "endpoint": endpoint, "method": method,
                                   "body": body, "rationale": rationale, "issue_id": issue_id},
                             timeout=KB_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": f"proposal failed: {exc}"}


# ---------------------------------------------------------------------------
# transparent proxy: /api/kb/** -> knowledge service
# ---------------------------------------------------------------------------
_PASSTHROUGH_HEADERS = {"content-type", "content-disposition", "content-length"}


@router.api_route("/api/kb/{path:path}",
                  methods=["GET", "POST", "PATCH", "DELETE", "PUT"])
async def kb_proxy(path: str, request: Request):
    url = f"{KB_URL}/api/kb/{path}"
    method = request.method
    # Uploads stream large PDFs; reindex_all can walk the whole corpus.
    timeout = KB_LONG_TIMEOUT if ("upload" in path or "reindex" in path) else KB_TIMEOUT

    try:
        body = await request.body()
        headers = {}
        if request.headers.get("content-type"):
            headers["Content-Type"] = request.headers["content-type"]

        # Run the blocking, fully-buffering requests call OFF the event loop, so a
        # large PDF upload (KB_LONG_TIMEOUT=300s) can't freeze every MVE endpoint.
        resp = await asyncio.to_thread(
            requests.request,
            method, url, params=dict(request.query_params),
            data=body if body else None, headers=headers, timeout=timeout,
        )

        out_headers = {k: v for k, v in resp.headers.items()
                       if k.lower() in _PASSTHROUGH_HEADERS}
        return Response(content=resp.content, status_code=resp.status_code,
                        headers=out_headers,
                        media_type=resp.headers.get("content-type"))
    except requests.exceptions.ConnectionError:
        _mark(False)
        return JSONResponse(
            status_code=503,
            content={"error": "knowledge_service_unavailable",
                     "detail": f"cannot reach the knowledge service at {KB_URL}. "
                               f"Start it with: docker compose -f docker-compose.yml "
                               f"-f knowledge_assistant/docker-compose.knowledge.yml up -d knowledge"},
        )
    except Exception as exc:
        logger.exception("MVE: knowledge proxy error on %s", path)
        return JSONResponse(status_code=502, content={"error": str(exc)})


@router.get("/api/knowledge/status")
def knowledge_status():
    """Consumed by the AI Assistant tab to enable/disable the knowledge toggle.
    Feeds the same liveness cache kb_available() uses, so the two never disagree."""
    try:
        resp = requests.get(f"{KB_URL}/health", timeout=5)
        resp.raise_for_status()
        _mark(True)
        return {"available": True, **resp.json()}
    except Exception as exc:
        _mark(False)
        return {"available": False, "error": str(exc), "url": KB_URL}
