"""Track AI usage per call (3.24.0).

Powers the per-site monthly billing the product team will hand to customers.
Every call to /api/why and /api/ai_query lands a row here with:
  - endpoint     ("/api/why" | "/api/ai_query")
  - mode         ("dot" | "class" | "score" | "verdict" | "chat")
  - tokens_in    (approx; computed from prompt length when provider doesn't tell us)
  - tokens_out   (approx; computed from answer length)
  - model_name   (the active AI model label, e.g. "Kimi", "gpt-4o")
  - provider     ("chatgpt" | "claude" | "gemini" | "local")
  - cost_usd     (estimated; based on the public Kimi rate card by default,
                  overridable per-model via audio_settings.ai.<name>.rate_*)
  - operator     (best-effort identifier — currently the X-Operator header or "anonymous")
  - latency_ms   (call wall time)
  - status       ("ok" | "error")
  - error        (free-text, populated on failures)

The table is idempotently created on first use; nothing to do during install.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default rate card — Kimi K2 via Moonshot list pricing (CNY → USD, early 2026).
# Per-model overrides live in audio_settings.ai.<name>.{rate_input_per_mtoken,
# rate_output_per_mtoken}, applied at log time.
DEFAULT_RATE_INPUT_PER_M  = 0.28   # USD per 1M input tokens
DEFAULT_RATE_OUTPUT_PER_M = 1.40   # USD per 1M output tokens


def _ensure_schema(conn) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_usage_log (
                time          TIMESTAMPTZ DEFAULT NOW(),
                endpoint      TEXT NOT NULL,
                mode          TEXT,
                model_name    TEXT,
                provider      TEXT,
                tokens_in     INTEGER DEFAULT 0,
                tokens_out    INTEGER DEFAULT 0,
                cost_usd      NUMERIC(10, 5) DEFAULT 0,
                latency_ms    INTEGER DEFAULT 0,
                status        TEXT DEFAULT 'ok',
                operator      TEXT,
                error         TEXT
            );
            CREATE INDEX IF NOT EXISTS ai_usage_log_time_idx ON ai_usage_log (time DESC);
            CREATE INDEX IF NOT EXISTS ai_usage_log_endpoint_idx ON ai_usage_log (endpoint);
            """
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.warning(f"ai_usage_log schema bootstrap failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def _approx_tokens(text: str) -> int:
    """Cheap token approximation (~4 chars per token for English/Persian/Arabic
    mixed text). Good enough for billing within ±20% — close enough since
    sub-cent costs round to zero anyway."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_cost(tokens_in: int, tokens_out: int,
                   rate_in: float = DEFAULT_RATE_INPUT_PER_M,
                   rate_out: float = DEFAULT_RATE_OUTPUT_PER_M) -> float:
    return round(
        (tokens_in  / 1_000_000.0) * rate_in
        + (tokens_out / 1_000_000.0) * rate_out,
        5,
    )


def log_usage(
    *,
    endpoint: str,
    mode: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_text: Optional[str] = None,
    answer_text: Optional[str] = None,
    latency_ms: int = 0,
    status: str = "ok",
    operator: Optional[str] = None,
    error: Optional[str] = None,
    rate_in: float = DEFAULT_RATE_INPUT_PER_M,
    rate_out: float = DEFAULT_RATE_OUTPUT_PER_M,
) -> None:
    """Fire-and-forget log. Never raises."""
    try:
        tokens_in  = _approx_tokens(prompt_text or "")
        tokens_out = _approx_tokens(answer_text or "")
        cost = _estimate_cost(tokens_in, tokens_out, rate_in, rate_out)
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return
        try:
            if not _ensure_schema(conn):
                return
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ai_usage_log
                  (endpoint, mode, model_name, provider, tokens_in, tokens_out,
                   cost_usd, latency_ms, status, operator, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (endpoint, mode, model_name, provider, tokens_in, tokens_out,
                 cost, latency_ms, status, operator, error),
            )
            conn.commit()
            cur.close()
        finally:
            try:
                release_db_connection(conn)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"ai_usage log skipped: {e}")


def summary(window: str = "30d") -> dict:
    """Roll-up for the billing dashboard.
    Returns {window, total_calls, total_cost_usd, by_endpoint:[…], by_day:[…]}."""
    _windows = {"24h": "24 hours", "7d": "7 days", "30d": "30 days", "90d": "90 days"}
    interval = _windows.get(window, "30 days")
    out = {"window": window, "total_calls": 0, "total_cost_usd": 0.0,
           "by_endpoint": [], "by_day": []}
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return out
        try:
            if not _ensure_schema(conn):
                return out
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) "
                "FROM ai_usage_log WHERE time > NOW() - INTERVAL %s",
                (interval,),
            )
            n, total = cur.fetchone()
            out["total_calls"] = int(n or 0)
            out["total_cost_usd"] = float(total or 0)

            cur.execute(
                "SELECT endpoint, COUNT(*) AS n, COALESCE(SUM(cost_usd), 0) AS cost "
                "FROM ai_usage_log WHERE time > NOW() - INTERVAL %s "
                "GROUP BY 1 ORDER BY n DESC",
                (interval,),
            )
            out["by_endpoint"] = [
                {"endpoint": row[0], "calls": int(row[1]), "cost_usd": float(row[2] or 0)}
                for row in cur.fetchall()
            ]

            cur.execute(
                "SELECT date_trunc('day', time) AS d, COUNT(*) AS n, "
                "       COALESCE(SUM(cost_usd), 0) AS cost "
                "FROM ai_usage_log WHERE time > NOW() - INTERVAL %s "
                "GROUP BY 1 ORDER BY 1",
                (interval,),
            )
            out["by_day"] = [
                {"day": row[0].strftime("%Y-%m-%d"),
                 "calls": int(row[1]), "cost_usd": float(row[2] or 0)}
                for row in cur.fetchall()
            ]
            cur.close()
        finally:
            try:
                release_db_connection(conn)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"ai_usage summary failed: {e}")
    return out
