"""Telegram + Bale bot delivery for shift reports (3.24.0).

Bale's bot API is deliberately Telegram-compatible at the request/response
level, so the same payload shape works for both — we just swap the base URL.
That means: one wrapper, two channels, almost the same code.

Usage:
    from services.messaging import send_document

    ok, info = send_document(
        channel="telegram",  # or "bale"
        token="...",
        chat_id="123456789",
        file_bytes=pdf_bytes,
        filename="shift_morning_2026-06-11.pdf",
        caption="Morning shift · Score 92.4 / 100 · RELEASE\\nReason: ...",
    )
    if not ok:
        logger.error(f"send failed: {info.get('error')}")
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, Dict, Any

import requests

logger = logging.getLogger(__name__)


# Channel → base URL. Both implement /sendDocument with identical shape.
_BASE_URLS = {
    "telegram": "https://api.telegram.org/bot{token}/",
    "bale":     "https://tapi.bale.ai/bot{token}/",
}


def _base(channel: str, token: str) -> str:
    template = _BASE_URLS.get(channel.lower())
    if not template:
        raise ValueError(f"unknown channel: {channel!r} (expected 'telegram' or 'bale')")
    return template.format(token=token)


def send_document(
    *,
    channel: str,
    token: str,
    chat_id: str,
    file_bytes: bytes,
    filename: str = "report.pdf",
    caption: Optional[str] = None,
    timeout: int = 60,
) -> Tuple[bool, Dict[str, Any]]:
    """Send a single PDF (or any bytes) to a Telegram/Bale chat.

    Returns (ok, info_dict). info_dict has either {message_id, chat_id}
    on success or {error, status_code, body} on failure.
    """
    if not token or not chat_id:
        return False, {"error": "token and chat_id are required"}
    url = _base(channel, token) + "sendDocument"
    data = {"chat_id": str(chat_id)}
    if caption:
        # Telegram + Bale both cap caption at ~1024 chars; truncate with an ellipsis
        # so we never lose the whole message just because the AI got verbose.
        if len(caption) > 1024:
            caption = caption[:1020] + "…"
        data["caption"] = caption
    files = {"document": (filename, file_bytes, "application/pdf")}
    try:
        r = requests.post(url, data=data, files=files, timeout=timeout)
    except requests.RequestException as e:
        return False, {"error": f"network error: {e}"}
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:400]}
    if r.status_code == 200 and isinstance(body, dict) and body.get("ok") is True:
        result = body.get("result") or {}
        return True, {
            "channel": channel,
            "message_id": result.get("message_id"),
            "chat_id": result.get("chat", {}).get("id") or chat_id,
            "sent_at": time.time(),
        }
    return False, {
        "channel": channel,
        "error": (body.get("description") if isinstance(body, dict) else None)
                 or f"HTTP {r.status_code}",
        "status_code": r.status_code,
        "body": body,
    }


def send_text(
    *,
    channel: str,
    token: str,
    chat_id: str,
    text: str,
    timeout: int = 30,
) -> Tuple[bool, Dict[str, Any]]:
    """Send a plain text message — used for quick health-check / test messages."""
    if not token or not chat_id:
        return False, {"error": "token and chat_id are required"}
    url = _base(channel, token) + "sendMessage"
    # Telegram limits to 4096 chars / Bale similar — truncate to be safe.
    if len(text) > 4000:
        text = text[:3990] + "…"
    try:
        r = requests.post(url, data={"chat_id": str(chat_id), "text": text}, timeout=timeout)
    except requests.RequestException as e:
        return False, {"error": f"network error: {e}"}
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:400]}
    if r.status_code == 200 and isinstance(body, dict) and body.get("ok") is True:
        result = body.get("result") or {}
        return True, {
            "channel": channel,
            "message_id": result.get("message_id"),
            "chat_id": result.get("chat", {}).get("id") or chat_id,
            "sent_at": time.time(),
        }
    return False, {
        "channel": channel,
        "error": (body.get("description") if isinstance(body, dict) else None)
                 or f"HTTP {r.status_code}",
        "status_code": r.status_code,
        "body": body,
    }


# ---------- Notification log ----------

def _ensure_log_schema(conn) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_log (
                time         TIMESTAMPTZ DEFAULT NOW(),
                channel      TEXT NOT NULL,         -- telegram / bale / email
                chat_id      TEXT,
                kind         TEXT,                  -- shift_report / test / why_summary
                caption      TEXT,
                status       TEXT,                  -- ok / error
                error        TEXT,
                latency_ms   INTEGER DEFAULT 0,
                schedule_name TEXT,
                schedule_cron TEXT
            );
            CREATE INDEX IF NOT EXISTS notification_log_time_idx ON notification_log (time DESC);
            """
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.warning(f"notification_log schema bootstrap failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def log_notification(
    *,
    channel: str,
    chat_id: Optional[str],
    kind: str,
    caption: Optional[str],
    status: str,
    error: Optional[str] = None,
    latency_ms: int = 0,
    schedule_name: Optional[str] = None,
    schedule_cron: Optional[str] = None,
) -> None:
    """Fire-and-forget notification audit. Never raises."""
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return
        try:
            if not _ensure_log_schema(conn):
                return
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO notification_log
                  (channel, chat_id, kind, caption, status, error, latency_ms,
                   schedule_name, schedule_cron)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (channel, chat_id, kind,
                 (caption or "")[:1000],
                 status, error, latency_ms, schedule_name, schedule_cron),
            )
            conn.commit()
            cur.close()
        finally:
            try:
                release_db_connection(conn)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"notification log skipped: {e}")


def recent_log(limit: int = 50) -> list:
    """Most recent notification log rows for the Admin → Notifications view."""
    out = []
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return out
        try:
            if not _ensure_log_schema(conn):
                return out
            cur = conn.cursor()
            cur.execute(
                """
                SELECT time, channel, chat_id, kind, status, error, latency_ms,
                       schedule_name, schedule_cron, caption
                FROM notification_log
                ORDER BY time DESC LIMIT %s
                """,
                (int(limit),),
            )
            for row in cur.fetchall():
                out.append({
                    "time": row[0].isoformat() if row[0] else None,
                    "channel": row[1],
                    "chat_id": row[2],
                    "kind": row[3],
                    "status": row[4],
                    "error": row[5],
                    "latency_ms": int(row[6] or 0),
                    "schedule_name": row[7],
                    "schedule_cron": row[8],
                    "caption": (row[9] or "")[:200],
                })
            cur.close()
        finally:
            try:
                release_db_connection(conn)
            except Exception:
                pass
    except Exception:
        pass
    return out
