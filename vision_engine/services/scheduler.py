"""Lightweight cron-style scheduler for shift report notifications (3.24.0).

Runs as a background asyncio task spawned at app startup. Wakes up once per
minute, checks each enabled `notifications.schedules[*]` entry against the
current local time, fires due jobs.

We support a tiny subset of cron expressions sufficient for shift schedules:
  - `*` wildcard
  - lists: `0,15,30,45`
  - simple ranges: `8-17`
  - step: `*/15`

Format: `MINUTE HOUR DOM MONTH DOW`. Five fields, space-separated. Same as
classic crontab. If your shift fires at 08:00, the cron is `0 8 * * *`.

Each schedule entry records `last_run` (UTC ISO timestamp) after a successful
fire, so we don't re-fire if the loop wakes twice in the same minute.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------- minimal cron parsing ----------

def _expand_field(field: str, lo: int, hi: int) -> set:
    """Expand a single cron field into the set of numeric matches."""
    field = (field or "").strip()
    if not field or field == "*":
        return set(range(lo, hi + 1))
    out = set()
    for part in field.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            base, step_s = part.split("/", 1)
            step = max(1, int(step_s))
            part = base or "*"
        if part == "*":
            rng = range(lo, hi + 1)
        elif "-" in part:
            a, b = part.split("-", 1)
            rng = range(max(lo, int(a)), min(hi, int(b)) + 1)
        else:
            v = int(part)
            rng = range(v, v + 1)
        for v in rng:
            if (v - lo) % step == 0:
                out.add(v)
    return out


def cron_matches(cron: str, now: datetime) -> bool:
    """Does `cron` (5-field) match this minute? Uses local time of `now`."""
    try:
        parts = cron.strip().split()
        if len(parts) != 5:
            return False
        m, h, dom, mon, dow = parts
        if now.minute not in _expand_field(m, 0, 59):
            return False
        if now.hour not in _expand_field(h, 0, 23):
            return False
        if now.day not in _expand_field(dom, 1, 31):
            return False
        if now.month not in _expand_field(mon, 1, 12):
            return False
        # Python's weekday() is Mon=0..Sun=6. Cron's DOW is Sun=0..Sat=6 traditionally,
        # but most modern implementations accept both (and Sun also = 7). Be permissive:
        py_dow = now.weekday()              # Mon=0..Sun=6
        cron_dow_sun0 = (py_dow + 1) % 7    # Sun=0..Sat=6
        wanted = _expand_field(dow, 0, 7)
        if not (py_dow in wanted or cron_dow_sun0 in wanted):
            return False
        return True
    except Exception:
        return False


# ---------- runner ----------

async def _fetch_shipment_report_pdf(app, shipment: str) -> Optional[bytes]:
    """Generate the shipment-quality PDF in-process by calling the existing
    handler. We pass a minimal Request shim — the handler reads
    `request.app.state` (3.21.18+).
    """
    try:
        # The actual exported name is `shipment_quality_score_report` (3.21.18).
        from routers.timeline import shipment_quality_score_report
        # FastAPI handlers take a Request; pass a shim that exposes `app`
        # because that's all the handler reads.
        class _FakeReq:
            def __init__(self, _app):
                self.app = _app
                self.headers = {}
        resp = await shipment_quality_score_report(_FakeReq(app), shipment=shipment, window="24h")
        # The handler returns a StreamingResponse for the PDF — iterate the body.
        if hasattr(resp, "body_iterator"):
            chunks = []
            async for chunk in resp.body_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                chunks.append(chunk)
            return b"".join(chunks)
        if hasattr(resp, "body"):
            return bytes(resp.body)
    except Exception as e:
        logger.warning(f"shift report PDF generation failed: {e}")
    return None


async def _fetch_why_for_score(app, shipment: str, language: str = "en") -> str:
    """Best-effort: ask /api/why mode=score for a 2-sentence caption pre-amble."""
    try:
        from routers.ai import why_for_dot
        import types
        # The handler reads request.app.state — fake a minimal Request shim.
        class _FakeReq:
            def __init__(self, app):
                self.app = app
                self.headers = {}
        fake_req = _FakeReq(app)
        # Pass minimal payload — handler fills in surrounding context itself.
        resp = await why_for_dot(fake_req, {
            "mode": "score",
            "metric": "shipment_quality_score",
            "shipment": shipment,
            "window_seconds": 3600,
            "language": language,
        })
        if hasattr(resp, "body"):
            import json as _json
            data = _json.loads(resp.body.decode())
            return (data.get("answer") or "").strip()
    except Exception as e:
        logger.debug(f"why caption fetch failed: {e}")
    return ""


async def run_one_schedule(schedule: Dict[str, Any], cfg: Dict[str, Any],
                            app, source: str = "cron") -> Tuple[bool, Dict[str, Any]]:
    """Generate the PDF + send via configured channels. Returns (overall_ok, info)."""
    from services.messaging import send_document, log_notification
    name = schedule.get("name") or "schedule"
    cron_s = schedule.get("cron") or ""
    channels = schedule.get("channels") or []
    chat_ids = [c for c in (schedule.get("chat_ids") or []) if c]
    include_why = bool(schedule.get("include_why", True))
    shipment = schedule.get("shipment_filter") or "no_shipment"

    pdf_bytes = await _fetch_shipment_report_pdf(app, shipment)
    if not pdf_bytes:
        info = {"error": "PDF generation failed"}
        for ch in channels:
            log_notification(
                channel=ch, chat_id=(chat_ids[0] if chat_ids else None),
                kind="shift_report", caption=name, status="error",
                error="PDF generation failed",
                schedule_name=name, schedule_cron=cron_s,
            )
        return False, info

    caption_lines = [f"Shift report · {name}"]
    if shipment and shipment != "no_shipment":
        caption_lines.append(f"Shipment: {shipment}")
    if include_why:
        why = await _fetch_why_for_score(app, shipment, "en")
        if why:
            caption_lines.append("")
            caption_lines.append("🤔 " + why)
    caption = "\n".join(caption_lines)

    overall_ok = False
    sends = []
    # 3.24.4 — only the "telegram" channel exists. Treat any incoming channel
    # name as telegram to stay backward-compatible with stored configs.
    ch = "telegram"
    ch_cfg = (cfg.get("channels") or {}).get(ch, {})
    token = ch_cfg.get("bot_token") or ""
    base_url = ch_cfg.get("base_url") or ""
    target_chat_ids = chat_ids or [ch_cfg.get("default_chat_id") or ""]
    for chat_id in target_chat_ids:
        if not (token and chat_id):
            log_notification(
                channel=ch, chat_id=chat_id or None,
                kind="shift_report", caption=name, status="error",
                error="missing token or chat_id",
                schedule_name=name, schedule_cron=cron_s,
            )
            continue
        import time as _t
        t0 = _t.time()
        ok, info = send_document(
            token=token, chat_id=chat_id, base_url=base_url,
            file_bytes=pdf_bytes,
            filename=f"shift_{name.replace(' ','_')}_{datetime.now().strftime('%Y-%m-%d_%H%M')}.pdf",
            caption=caption,
        )
        overall_ok = overall_ok or ok
        sends.append({"channel": ch, "chat_id": chat_id, "ok": ok, "info": info})
        log_notification(
            channel=ch, chat_id=chat_id, kind="shift_report",
            caption=caption, status=("ok" if ok else "error"),
            error=(None if ok else str(info.get("error"))),
            latency_ms=int((_t.time() - t0) * 1000),
            schedule_name=name, schedule_cron=cron_s,
        )
    return overall_ok, {"source": source, "sends": sends}


# ---------- background loop ----------

_scheduler_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None


async def _scheduler_loop(app):
    """Wake every minute, check each enabled schedule, fire matching ones."""
    global _stop_event
    _stop_event = asyncio.Event()
    logger.info("notifications scheduler started")
    while not _stop_event.is_set():
        try:
            from config import load_service_config
            svc = load_service_config() or {}
            cfg = svc.get("notifications") or {}
            schedules = cfg.get("schedules") or []
            now = datetime.now()  # local time
            for sched in schedules:
                if not sched.get("enabled"):
                    continue
                cron_s = sched.get("cron") or ""
                if not cron_matches(cron_s, now):
                    continue
                last_run = sched.get("last_run")
                this_minute = now.strftime("%Y-%m-%dT%H:%M")
                if last_run and last_run.startswith(this_minute):
                    continue  # already fired this minute
                logger.info(f"notifications: firing {sched.get('name')!r} (cron={cron_s})")
                try:
                    ok, info = await run_one_schedule(sched, cfg, app, source="cron")
                    sched["last_run"] = now.strftime("%Y-%m-%dT%H:%M:%S")
                    sched["last_status"] = "ok" if ok else "error"
                except Exception as e:
                    logger.warning(f"schedule {sched.get('name')!r} fire failed: {e}")
                    sched["last_run"] = now.strftime("%Y-%m-%dT%H:%M:%S")
                    sched["last_status"] = "error"
            # Persist any last_run updates back to the config so reboots don't replay.
            if schedules:
                from config import save_service_config
                svc["notifications"] = cfg
                try:
                    save_service_config(svc)
                except Exception as _e:
                    logger.debug(f"save_service_config after scheduler tick failed: {_e}")
        except Exception as e:
            logger.warning(f"scheduler tick error: {e}")
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass
    logger.info("notifications scheduler stopped")


def start_scheduler(app):
    """Spawn the scheduler task. Idempotent — second call is a no-op."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        return _scheduler_task
    loop = asyncio.get_event_loop()
    _scheduler_task = loop.create_task(_scheduler_loop(app))
    return _scheduler_task


def stop_scheduler():
    """Signal the loop to exit. Caller awaits the task separately."""
    if _stop_event is not None:
        _stop_event.set()
