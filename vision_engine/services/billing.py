"""Billing / subscription / plans / pluggable payment gateways — 4.0.362.

Brings the Monitait console's paid-product model on-prem, adapted for a single
factory. Four capabilities the operator asked for:

  1. Paid API access        — issue API keys, gate them on the plan, meter usage.
  2. Monthly subscription   — a plan with a billing period + gateway payment.
  3. Plans                  — basic / economy / enterprise, each with limits
                              (data-RETENTION being the headline: basic = 7 days).
  4. On-site service request— ask for a periodic technician visit.

Two hard safety rules, because this touches money and data deletion:

  * SAFE BY DEFAULT. A fresh install is seeded on the **enterprise (unlimited)**
    plan, so retention pruning and feature gating are INERT until an admin
    explicitly subscribes to a limited plan. Nothing is deleted or blocked out of
    the box.

  * PLUGGABLE, ENV-DEFINED GATEWAYS. The active gateway is chosen by
    `PAYMENT_PROVIDER` (e.g. zarinpal / idpay / paypal / manual) and its
    credentials come from env. With nothing configured the **manual** gateway is
    used — an offline invoice the admin marks paid — so the whole flow works with
    no external dependency (important in a disconnected / sanctioned network).

Storage is MVE's TimescaleDB (mve_plans / mve_subscription / mve_payments /
mve_api_keys / mve_service_requests), self-bootstrapping.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional

from services.db import get_db_connection, release_db_connection

logger = logging.getLogger(__name__)

_schema_ready = False

# Default plan catalogue. retention_days=None means UNLIMITED. Prices are a hint —
# the operator edits them (or the gateway amount) per install; currency too.
_DEFAULT_PLANS = [
    # code,       name,         retention_days, max_machines, api,   service, price, currency, period_days
    ("basic",      "Basic",       7,              2,           False, False,   0,     "USD",    30),
    ("economy",    "Economy",     90,             20,          True,  False,   0,     "USD",    30),
    ("enterprise", "Enterprise",  None,           None,        True,  True,    0,     "USD",    30),
]
DEFAULT_PLAN = "enterprise"   # fresh install = unlimited, nothing pruned/gated


def _exec(sql: str, params: tuple = (), fetch: Optional[str] = None):
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("db unavailable")
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        out = cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None
        conn.commit()
        cur.close()
        return out
    finally:
        release_db_connection(conn)


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_plans (
            code           TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            retention_days INTEGER,            -- NULL = unlimited
            max_machines   INTEGER,            -- NULL = unlimited
            api_enabled    BOOLEAN DEFAULT FALSE,
            service_incl   BOOLEAN DEFAULT FALSE,
            price          DOUBLE PRECISION DEFAULT 0,
            currency       TEXT DEFAULT 'USD',
            period_days    INTEGER DEFAULT 30,
            features       JSONB
        );""")
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_subscription (
            id          BIGSERIAL PRIMARY KEY,
            plan_code   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'active',   -- active | pending | expired
            gateway     TEXT,
            payment_ref TEXT,
            started_at  TIMESTAMPTZ DEFAULT NOW(),
            expires_at  TIMESTAMPTZ
        );""")
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_payments (
            id         BIGSERIAL PRIMARY KEY,
            ref        TEXT UNIQUE NOT NULL,
            gateway    TEXT,
            plan_code  TEXT,
            amount     DOUBLE PRECISION,
            currency   TEXT,
            status     TEXT DEFAULT 'pending',            -- pending | paid | failed
            redirect   TEXT,
            meta       JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            paid_at    TIMESTAMPTZ
        );""")
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_api_keys (
            api_key    TEXT PRIMARY KEY,
            label      TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_used  TIMESTAMPTZ,
            calls      BIGINT DEFAULT 0,
            revoked    BOOLEAN DEFAULT FALSE
        );""")
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_service_requests (
            id         BIGSERIAL PRIMARY KEY,
            kind       TEXT DEFAULT 'periodic_service',   -- periodic_service | repair | install | other
            note       TEXT,
            contact    TEXT,
            status     TEXT DEFAULT 'open',               -- open | scheduled | done | cancelled
            created_at TIMESTAMPTZ DEFAULT NOW()
        );""")
    # seed the plan catalogue (idempotent — keep operator edits, only add missing)
    for (code, name, ret, mx, api, svc, price, cur, period) in _DEFAULT_PLANS:
        _exec("INSERT INTO mve_plans (code, name, retention_days, max_machines, api_enabled, "
              "service_incl, price, currency, period_days) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
              "ON CONFLICT (code) DO NOTHING",
              (code, name, ret, mx, api, svc, price, cur, period))
    # seed a default subscription = enterprise (unlimited) so nothing is ever pruned
    row = _exec("SELECT COUNT(*) FROM mve_subscription", fetch="one")
    if not row or int(row[0] or 0) == 0:
        _exec("INSERT INTO mve_subscription (plan_code, status, gateway) VALUES (%s,'active','seed')",
              (DEFAULT_PLAN,))
        logger.info("billing: seeded default subscription = %s (unlimited, safe)", DEFAULT_PLAN)
    _schema_ready = True


# --------------------------------------------------------------------------- #
# plans + subscription
# --------------------------------------------------------------------------- #
def list_plans() -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT code, name, retention_days, max_machines, api_enabled, service_incl, "
                 "price, currency, period_days FROM mve_plans ORDER BY "
                 "COALESCE(retention_days, 2147483647)", fetch="all") or []
    return [{"code": r[0], "name": r[1], "retention_days": r[2], "max_machines": r[3],
             "api_enabled": r[4], "service_incl": r[5], "price": r[6], "currency": r[7],
             "period_days": r[8]} for r in rows]


def get_plan(code: str) -> Optional[Dict[str, Any]]:
    for p in list_plans():
        if p["code"] == code:
            return p
    return None


def current_subscription() -> Dict[str, Any]:
    _ensure_schema()
    row = _exec("SELECT plan_code, status, gateway, payment_ref, started_at, expires_at "
                "FROM mve_subscription WHERE status='active' ORDER BY id DESC LIMIT 1", fetch="one")
    if not row:
        return {"plan_code": DEFAULT_PLAN, "status": "active", "plan": get_plan(DEFAULT_PLAN)}
    return {"plan_code": row[0], "status": row[1], "gateway": row[2], "payment_ref": row[3],
            "started_at": str(row[4]) if row[4] else None,
            "expires_at": str(row[5]) if row[5] else None, "plan": get_plan(row[0])}


def _activate_subscription(plan_code: str, gateway: str, ref: Optional[str]) -> None:
    """Make plan_code the single active subscription."""
    plan = get_plan(plan_code)
    period = (plan or {}).get("period_days") or 30
    _exec("UPDATE mve_subscription SET status='expired' WHERE status='active'")
    _exec("INSERT INTO mve_subscription (plan_code, status, gateway, payment_ref, expires_at) "
          "VALUES (%s,'active',%s,%s, NOW() + (%s || ' days')::interval)",
          (plan_code, gateway, ref, str(int(period))))


# --------------------------------------------------------------------------- #
# pluggable payment gateways (env-defined)
# --------------------------------------------------------------------------- #
def active_provider() -> str:
    return (os.environ.get("PAYMENT_PROVIDER") or "manual").strip().lower()


def gateway_status() -> Dict[str, Any]:
    prov = active_provider()
    drv = _GATEWAYS.get(prov)
    return {"provider": prov, "known": drv is not None,
            "configured": bool(drv and drv.configured()),
            "available": sorted(_GATEWAYS.keys())}


class PaymentGateway:
    """Driver contract. create_payment returns {ref, redirect, status}; verify
    returns {ok, paid, amount}. Drivers read their own creds from env."""
    name = "base"

    def configured(self) -> bool:
        return True

    def create_payment(self, amount: float, currency: str, description: str,
                       return_url: str, ref: str) -> Dict[str, Any]:
        raise NotImplementedError

    def verify_payment(self, ref: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ManualGateway(PaymentGateway):
    """No external call — the payment is an offline invoice an admin marks paid.
    The always-available fallback so billing works with no gateway configured."""
    name = "manual"

    def create_payment(self, amount, currency, description, return_url, ref):
        return {"ref": ref, "redirect": None, "status": "manual_pending"}

    def verify_payment(self, ref, params):
        # A manual payment is confirmed by the admin endpoint, not a gateway call.
        return {"ok": True, "paid": False, "amount": None, "manual": True}


class _EnvHttpGateway(PaymentGateway):
    """Base for real gateways: reads creds from env, makes the HTTP call when
    `requests` + creds are present, and reports `configured()` honestly so the UI
    can show what still needs setup. Concrete request/verify per provider."""
    env_keys: tuple = ()

    def _cfg(self, key: str) -> str:
        return os.environ.get(key, "").strip()

    def configured(self) -> bool:
        return all(self._cfg(k) for k in self.env_keys)


class ZarinpalGateway(_EnvHttpGateway):
    name = "zarinpal"
    env_keys = ("PAYMENT_ZARINPAL_MERCHANT",)

    def create_payment(self, amount, currency, description, return_url, ref):
        if not self.configured():
            return {"ref": ref, "redirect": None, "status": "unconfigured"}
        import requests
        merchant = self._cfg("PAYMENT_ZARINPAL_MERCHANT")
        r = requests.post("https://api.zarinpal.com/pg/v4/payment/request.json",
                          json={"merchant_id": merchant, "amount": int(amount),
                                "description": description, "callback_url": return_url},
                          timeout=20).json()
        authority = (r.get("data") or {}).get("authority")
        if not authority:
            return {"ref": ref, "redirect": None, "status": "failed", "detail": r}
        return {"ref": authority, "redirect": f"https://www.zarinpal.com/pg/StartPay/{authority}",
                "status": "redirect"}

    def verify_payment(self, ref, params):
        import requests
        merchant = self._cfg("PAYMENT_ZARINPAL_MERCHANT")
        amount = int(params.get("amount") or 0)
        r = requests.post("https://api.zarinpal.com/pg/v4/payment/verify.json",
                          json={"merchant_id": merchant, "amount": amount, "authority": ref},
                          timeout=20).json()
        code = (r.get("data") or {}).get("code")
        return {"ok": code in (100, 101), "paid": code in (100, 101), "amount": amount}


class IdpayGateway(_EnvHttpGateway):
    name = "idpay"
    env_keys = ("PAYMENT_IDPAY_API_KEY",)

    def create_payment(self, amount, currency, description, return_url, ref):
        if not self.configured():
            return {"ref": ref, "redirect": None, "status": "unconfigured"}
        import requests
        r = requests.post("https://api.idpay.ir/v1.1/payment",
                          headers={"X-API-KEY": self._cfg("PAYMENT_IDPAY_API_KEY"),
                                   "Content-Type": "application/json"},
                          json={"order_id": ref, "amount": int(amount),
                                "callback": return_url, "desc": description},
                          timeout=20).json()
        link = r.get("link")
        if not link:
            return {"ref": ref, "redirect": None, "status": "failed", "detail": r}
        return {"ref": r.get("id", ref), "redirect": link, "status": "redirect"}

    def verify_payment(self, ref, params):
        import requests
        r = requests.post("https://api.idpay.ir/v1.1/payment/verify",
                          headers={"X-API-KEY": self._cfg("PAYMENT_IDPAY_API_KEY"),
                                   "Content-Type": "application/json"},
                          json={"id": ref, "order_id": params.get("order_id", ref)},
                          timeout=20).json()
        return {"ok": str(r.get("status")) in ("100", "200"),
                "paid": str(r.get("status")) in ("100", "200"), "amount": r.get("amount")}


class PaypalGateway(_EnvHttpGateway):
    name = "paypal"
    env_keys = ("PAYMENT_PAYPAL_CLIENT_ID", "PAYMENT_PAYPAL_SECRET")

    def _base(self) -> str:
        return ("https://api-m.paypal.com" if self._cfg("PAYMENT_PAYPAL_LIVE")
                else "https://api-m.sandbox.paypal.com")

    def _token(self):
        import requests
        r = requests.post(f"{self._base()}/v1/oauth2/token",
                          auth=(self._cfg("PAYMENT_PAYPAL_CLIENT_ID"), self._cfg("PAYMENT_PAYPAL_SECRET")),
                          data={"grant_type": "client_credentials"}, timeout=20)
        return r.json().get("access_token")

    def create_payment(self, amount, currency, description, return_url, ref):
        if not self.configured():
            return {"ref": ref, "redirect": None, "status": "unconfigured"}
        import requests
        tok = self._token()
        r = requests.post(f"{self._base()}/v2/checkout/orders",
                          headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                          json={"intent": "CAPTURE",
                                "purchase_units": [{"amount": {"currency_code": currency or "USD",
                                                               "value": f"{amount:.2f}"}, "description": description}],
                                "application_context": {"return_url": return_url}},
                          timeout=20).json()
        approve = next((l["href"] for l in r.get("links", []) if l.get("rel") == "approve"), None)
        return {"ref": r.get("id", ref), "redirect": approve,
                "status": "redirect" if approve else "failed"}

    def verify_payment(self, ref, params):
        import requests
        tok = self._token()
        r = requests.post(f"{self._base()}/v2/checkout/orders/{ref}/capture",
                          headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                          timeout=20).json()
        ok = r.get("status") == "COMPLETED"
        return {"ok": ok, "paid": ok, "amount": None}


# driver registry — add a class here (or an install-specific one) and it's selectable by env
_GATEWAYS: Dict[str, PaymentGateway] = {
    g.name: g for g in (ManualGateway(), ZarinpalGateway(), IdpayGateway(), PaypalGateway())
}


def _gateway() -> PaymentGateway:
    return _GATEWAYS.get(active_provider(), _GATEWAYS["manual"])


# --------------------------------------------------------------------------- #
# subscribe flow
# --------------------------------------------------------------------------- #
def subscribe(plan_code: str, return_url: str = "") -> Dict[str, Any]:
    """Start a subscription to plan_code. Creates a payment via the active gateway
    and returns a redirect (real gateway) or a manual-pending invoice. Free plans
    (price 0) activate immediately."""
    _ensure_schema()
    plan = get_plan(plan_code)
    if not plan:
        raise ValueError(f"unknown plan: {plan_code}")
    amount = float(plan.get("price") or 0)
    currency = plan.get("currency") or "USD"

    if amount <= 0:
        _activate_subscription(plan_code, "free", None)
        return {"status": "active", "plan_code": plan_code, "free": True}

    gw = _gateway()
    ref = "pay_" + secrets.token_urlsafe(12)
    res = gw.create_payment(amount, currency, f"MVE {plan['name']} subscription",
                            return_url, ref)
    real_ref = res.get("ref", ref)
    _exec("INSERT INTO mve_payments (ref, gateway, plan_code, amount, currency, status, redirect) "
          "VALUES (%s,%s,%s,%s,%s,%s,%s)",
          (real_ref, gw.name, plan_code, amount, currency,
           "pending", res.get("redirect")))
    return {"status": res.get("status", "pending"), "gateway": gw.name,
            "ref": real_ref, "redirect": res.get("redirect"),
            "plan_code": plan_code, "amount": amount, "currency": currency}


def confirm_payment(ref: str, params: Optional[Dict[str, Any]] = None,
                    manual: bool = False) -> Dict[str, Any]:
    """Verify a payment (gateway callback) or mark a manual invoice paid; on
    success, activate the subscription."""
    _ensure_schema()
    row = _exec("SELECT gateway, plan_code, amount, status FROM mve_payments WHERE ref=%s",
                (ref,), fetch="one")
    if not row:
        raise ValueError("unknown payment ref")
    gateway, plan_code, amount, status = row
    if status == "paid":
        return {"ok": True, "already": True, "plan_code": plan_code}
    paid = manual
    if not manual:
        gw = _GATEWAYS.get(gateway, _GATEWAYS["manual"])
        verify = gw.verify_payment(ref, {**(params or {}), "amount": amount})
        paid = bool(verify.get("paid"))
    if paid:
        _exec("UPDATE mve_payments SET status='paid', paid_at=NOW() WHERE ref=%s", (ref,))
        _activate_subscription(plan_code, gateway, ref)
        return {"ok": True, "paid": True, "plan_code": plan_code}
    return {"ok": False, "paid": False, "plan_code": plan_code}


def list_payments(limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT ref, gateway, plan_code, amount, currency, status, created_at, paid_at "
                 "FROM mve_payments ORDER BY id DESC LIMIT %s", (int(limit),), fetch="all") or []
    return [{"ref": r[0], "gateway": r[1], "plan_code": r[2], "amount": r[3], "currency": r[4],
             "status": r[5], "created_at": str(r[6]), "paid_at": str(r[7]) if r[7] else None}
            for r in rows]


# --------------------------------------------------------------------------- #
# API keys (paid API access + metering)
# --------------------------------------------------------------------------- #
def api_enabled() -> bool:
    plan = current_subscription().get("plan") or {}
    return bool(plan.get("api_enabled"))


def create_api_key(label: str = "") -> Dict[str, Any]:
    _ensure_schema()
    key = "mve_" + secrets.token_urlsafe(24)
    _exec("INSERT INTO mve_api_keys (api_key, label) VALUES (%s,%s)", (key, label.strip() or None))
    return {"api_key": key, "label": label.strip() or None, "api_enabled": api_enabled()}


def list_api_keys() -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT api_key, label, created_at, last_used, calls, revoked "
                 "FROM mve_api_keys ORDER BY created_at DESC", fetch="all") or []
    def _mask(k):
        return (k[:8] + "…" + k[-4:]) if k and len(k) > 14 else k
    return [{"api_key_masked": _mask(r[0]), "label": r[1], "created_at": str(r[2]),
             "last_used": str(r[3]) if r[3] else None, "calls": int(r[4] or 0),
             "revoked": r[5]} for r in rows]


def revoke_api_key(masked_or_key: str) -> None:
    _ensure_schema()
    # accept either the full key or the prefix shown in the UI
    prefix = masked_or_key.split("…")[0]
    _exec("UPDATE mve_api_keys SET revoked=TRUE WHERE api_key=%s OR api_key LIKE %s",
          (masked_or_key, prefix + "%"))


def verify_api_key(key: str) -> bool:
    """True if the key exists, isn't revoked, AND the plan includes API access.
    Meters the call. Used by an API-auth dependency (wired separately)."""
    _ensure_schema()
    if not key:
        return False
    row = _exec("SELECT revoked FROM mve_api_keys WHERE api_key=%s", (key,), fetch="one")
    if not row or row[0]:
        return False
    if not api_enabled():
        return False
    _exec("UPDATE mve_api_keys SET calls=calls+1, last_used=NOW() WHERE api_key=%s", (key,))
    return True


# --------------------------------------------------------------------------- #
# on-site service requests
# --------------------------------------------------------------------------- #
def create_service_request(kind: str = "periodic_service", note: str = "",
                           contact: str = "") -> Dict[str, Any]:
    _ensure_schema()
    row = _exec("INSERT INTO mve_service_requests (kind, note, contact) VALUES (%s,%s,%s) RETURNING id",
                (kind or "periodic_service", note.strip() or None, contact.strip() or None),
                fetch="one")
    return {"ok": True, "id": int(row[0]) if row else None}


def list_service_requests(limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT id, kind, note, contact, status, created_at FROM mve_service_requests "
                 "ORDER BY id DESC LIMIT %s", (int(limit),), fetch="all") or []
    return [{"id": r[0], "kind": r[1], "note": r[2], "contact": r[3], "status": r[4],
             "created_at": str(r[5])} for r in rows]


def set_service_request_status(req_id: int, status: str) -> None:
    _ensure_schema()
    _exec("UPDATE mve_service_requests SET status=%s WHERE id=%s", (status, int(req_id)))


# --------------------------------------------------------------------------- #
# data retention (the headline plan limit) — DESTRUCTIVE, so gated hard
# --------------------------------------------------------------------------- #
def active_retention_days() -> Optional[int]:
    """Retention window of the active plan; None = unlimited (never prune)."""
    plan = current_subscription().get("plan") or {}
    rd = plan.get("retention_days")
    return int(rd) if rd else None


# Which tables retention prunes. Keyed table -> timestamp column. Deliberately
# ONLY watcher_metrics by default — the operator opts real inspection data in via
# env if they truly want it pruned.
_RETENTION_TABLES = {"watcher_metrics": "ts"}


def enforce_retention(dry_run: bool = True) -> Dict[str, Any]:
    """Delete rows older than the active plan's retention window.

    GATED THREE WAYS so it can never surprise-delete: (1) no-op unless a finite
    retention plan is active, (2) `dry_run=True` by default only counts, (3) live
    pruning requires env `BILLING_RETENTION_ENFORCE=1`. Returns per-table counts."""
    _ensure_schema()
    days = active_retention_days()
    if not days:
        return {"enforced": False, "reason": "unlimited plan — nothing to prune"}
    live = (not dry_run) and os.environ.get("BILLING_RETENTION_ENFORCE", "").lower() in ("1", "true", "yes")
    result: Dict[str, Any] = {"retention_days": days, "live": live, "tables": {}}
    for table, tscol in _RETENTION_TABLES.items():
        try:
            cnt = _exec(f"SELECT COUNT(*) FROM {table} WHERE {tscol} < NOW() - (%s || ' days')::interval",
                        (str(days),), fetch="one")
            n = int(cnt[0]) if cnt else 0
            if live and n:
                _exec(f"DELETE FROM {table} WHERE {tscol} < NOW() - (%s || ' days')::interval", (str(days),))
            result["tables"][table] = {"eligible": n, "deleted": (n if live else 0)}
        except Exception as e:
            result["tables"][table] = {"error": str(e)}
    return result
