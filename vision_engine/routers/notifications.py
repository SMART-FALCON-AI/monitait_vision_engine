"""Notifications + AI-usage admin endpoints (3.24.0).

Config shape (lives at service_config.notifications):
{
  "channels": {
    "telegram": {"enabled": false, "bot_token": "", "default_chat_id": ""},
    "bale":     {"enabled": false, "bot_token": "", "default_chat_id": ""}
  },
  "schedules": [
    {
      "name": "End of morning shift",
      "cron": "0 8 * * *",
      "channels": ["telegram", "bale"],
      "chat_ids": ["123456789"],        // overrides default if non-empty
      "include_why": true,
      "shipment_filter": "",            // empty = current/most-recent shipment
      "enabled": true,
      "last_run": null,                 // ISO timestamp, populated by scheduler
      "last_status": null               // "ok" | "error" | null
    },
    ...
  ]
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests as _requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import load_service_config, save_service_config


logger = logging.getLogger(__name__)
router = APIRouter()


def _default_channels() -> Dict[str, Any]:
    # 3.24.4 — single "telegram" channel. The base_url is configurable so the
    # operator can point at any Telegram-compatible bot host without code
    # changes (Bale, self-hosted gateway, custom relay, etc.). Default empty
    # base_url means "use the standard Telegram API host".
    return {
        "telegram": {
            "enabled": False, "bot_token": "", "default_chat_id": "",
            "base_url": "",
        },
    }


def _get_config() -> Dict[str, Any]:
    svc = load_service_config() or {}
    cfg = (svc.get("notifications") or {})
    if "channels" not in cfg:
        cfg["channels"] = _default_channels()
    if "schedules" not in cfg or not isinstance(cfg["schedules"], list):
        cfg["schedules"] = []
    return cfg


def _save_config(cfg: Dict[str, Any]) -> None:
    svc = load_service_config() or {}
    svc["notifications"] = cfg
    save_service_config(svc)


def _mask_token(t: Optional[str]) -> str:
    if not t:
        return ""
    if len(t) <= 8:
        return "***"
    return t[:4] + "…" + t[-4:]


@router.get("/api/notifications/config")
async def get_notifications_config():
    """Return the notification config with bot tokens MASKED.
    The full token is only visible to scheduled jobs and test-send (server-side)."""
    cfg = _get_config()
    # 3.24.4 — silently drop any legacy "bale" key still in storage so the
    # UI / clients never see it after the next config save.
    if "bale" in (cfg.get("channels") or {}):
        cfg["channels"].pop("bale", None)
    safe = {"channels": {}, "schedules": cfg.get("schedules", [])}
    for ch, val in (cfg.get("channels") or {}).items():
        safe["channels"][ch] = {
            "enabled": bool(val.get("enabled", False)),
            "bot_token_masked": _mask_token(val.get("bot_token", "")),
            "default_chat_id": val.get("default_chat_id", ""),
            "base_url": val.get("base_url", ""),
        }
    return JSONResponse(content=safe)


@router.post("/api/notifications/config")
async def post_notifications_config(payload: Dict[str, Any]):
    """Update notification config.

    Payload accepts:
      - {channel: 'telegram'|'bale', bot_token?, default_chat_id?, enabled?}
        — partial update of one channel
      - {schedules: [...]}  — replace the schedule list wholesale
    """
    cfg = _get_config()

    if "channel" in payload:
        ch = str(payload["channel"]).lower()
        if ch != "telegram":
            return JSONResponse(content={"error": "only 'telegram' channel is supported"}, status_code=400)
        entry = cfg["channels"].setdefault(ch, {"enabled": False, "bot_token": "", "default_chat_id": "", "base_url": ""})
        if "bot_token" in payload and payload["bot_token"]:  # only overwrite if non-empty
            entry["bot_token"] = str(payload["bot_token"])
        if "default_chat_id" in payload:
            entry["default_chat_id"] = str(payload["default_chat_id"] or "")
        if "base_url" in payload:
            entry["base_url"] = str(payload["base_url"] or "").strip()
        if "enabled" in payload:
            entry["enabled"] = bool(payload["enabled"])
    # 3.24.4 — clean up any legacy bale entry whenever we save.
    cfg["channels"].pop("bale", None)

    if "schedules" in payload and isinstance(payload["schedules"], list):
        clean = []
        for s in payload["schedules"]:
            if not isinstance(s, dict):
                continue
            # 3.24.4 — schedules don't pick channels anymore (only telegram exists).
            # Strip any incoming legacy values and write the canonical single-channel form.
            clean.append({
                "name": str(s.get("name") or ""),
                "cron": str(s.get("cron") or ""),
                "channels": ["telegram"],
                "chat_ids": [str(x) for x in (s.get("chat_ids") or []) if str(x).strip()],
                "include_why": bool(s.get("include_why", True)),
                "shipment_filter": str(s.get("shipment_filter") or ""),
                "enabled": bool(s.get("enabled", True)),
                "last_run": s.get("last_run"),
                "last_status": s.get("last_status"),
            })
        cfg["schedules"] = clean

    _save_config(cfg)
    return JSONResponse(content={"success": True})


@router.post("/api/notifications/test_send")
async def test_send(payload: Dict[str, Any]):
    """Send a tiny test message to the configured chat to validate token+chat_id."""
    cfg = _get_config()
    channel = "telegram"  # 3.24.4 — only one supported channel
    entry = (cfg.get("channels") or {}).get(channel, {})
    token = entry.get("bot_token") or ""
    chat_id = str(payload.get("chat_id") or entry.get("default_chat_id") or "")
    base_url = entry.get("base_url") or ""
    if not token or not chat_id:
        return JSONResponse(content={"error": "bot_token and chat_id are required (save the channel config first)"},
                            status_code=400)

    from services.messaging import send_text, log_notification
    import time as _t
    _t0 = _t.time()
    ok, info = send_text(
        token=token, chat_id=chat_id, base_url=base_url,
        text="[MVE test] ✓ This channel is wired up correctly.",
    )
    log_notification(
        channel=channel, chat_id=chat_id, kind="test",
        caption="[MVE test]", status=("ok" if ok else "error"),
        error=(None if ok else str(info.get("error"))),
        latency_ms=int((_t.time() - _t0) * 1000),
    )
    if ok:
        return JSONResponse(content={"success": True, "info": info})
    return JSONResponse(content={"success": False, "info": info}, status_code=502)


@router.post("/api/notifications/send_now")
async def send_now(payload: Dict[str, Any], request: Request):
    """Manually trigger a shift-report send (skips the cron). Used by the
    'Send now' button in the Notifications panel."""
    from services.scheduler import run_one_schedule
    schedule = payload.get("schedule")  # an inline schedule object OR null = a one-shot
    cfg = _get_config()
    if not schedule:
        # Build a one-shot from the top-level config (telegram is the only channel)
        ch = (cfg.get("channels") or {}).get("telegram", {})
        schedule = {
            "name": "manual",
            "cron": "",
            "channels": ["telegram"],
            "chat_ids": [str(payload.get("chat_id") or ch.get("default_chat_id") or "")],
            "include_why": bool(payload.get("include_why", True)),
            "shipment_filter": str(payload.get("shipment_filter") or ""),
            "enabled": True,
        }
    ok, info = await run_one_schedule(schedule, cfg, request.app, source="manual")
    return JSONResponse(content={"success": ok, "info": info})


@router.get("/api/notifications/log")
async def notifications_log(limit: int = 50):
    """Recent notification audit rows. Powers the log table on the
    Advanced → Notifications panel."""
    from services.messaging import recent_log
    return JSONResponse(content={"items": recent_log(limit)})


# ---------- AI usage admin ----------

@router.get("/api/ai_usage/summary")
async def ai_usage_summary(window: str = "30d"):
    """Roll-up of AI usage for the billing dashboard."""
    from services.ai_usage import summary as _sum
    return JSONResponse(content=_sum(window))
