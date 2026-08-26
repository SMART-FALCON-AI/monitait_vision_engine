"""Billing / subscription / plans / payment / API-keys / service-requests API
(4.0.362). Thin HTTP layer over services/billing.py.

Registered in main.py before commands_router. Safe by default: a fresh install is
on the unlimited plan, so nothing here prunes data or blocks a feature until an
admin subscribes to a limited plan. The payment gateway is pluggable and chosen by
the PAYMENT_PROVIDER env var; with none set the 'manual' (invoice) flow is used.
"""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from services import billing

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# plans + subscription
# --------------------------------------------------------------------------- #
@router.get("/api/billing/plans")
def plans():
    return {"plans": billing.list_plans()}


@router.get("/api/billing/subscription")
def subscription():
    return billing.current_subscription()


@router.get("/api/billing/gateway")
def gateway():
    """Which payment provider is active + whether its env creds are configured."""
    return billing.gateway_status()


@router.post("/api/billing/subscribe")
async def subscribe(request: Request):
    try:
        b = await request.json()
        return billing.subscribe(str(b.get("plan_code", "")), str(b.get("return_url", "")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.warning("subscribe failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/billing/confirm")
async def confirm(request: Request):
    """Verify a gateway payment, or (manual=true) mark an offline invoice paid."""
    try:
        b = await request.json()
        return billing.confirm_payment(str(b.get("ref", "")), b.get("params") or {},
                                       manual=bool(b.get("manual")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/billing/callback")
async def callback(request: Request):
    """Public gateway return URL — the user's browser is redirected here after
    paying. Verifies, then bounces to the dashboard. Never gated (a GET)."""
    params = dict(request.query_params)
    ref = params.get("ref") or params.get("authority") or params.get("order_id") or ""
    ok = False
    try:
        if ref:
            ok = bool(billing.confirm_payment(ref, params).get("paid"))
    except Exception as e:
        logger.warning("billing callback verify failed: %s", e)
    return RedirectResponse(url=f"/status#billing?paid={'1' if ok else '0'}")


@router.get("/api/billing/payments")
def payments():
    return {"payments": billing.list_payments()}


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #
@router.get("/api/billing/api-keys")
def api_keys():
    return {"keys": billing.list_api_keys(), "api_enabled": billing.api_enabled()}


@router.post("/api/billing/api-keys")
async def add_api_key(request: Request):
    b = await request.json()
    return billing.create_api_key(str(b.get("label", "")))


@router.delete("/api/billing/api-keys/{key}")
def del_api_key(key: str):
    billing.revoke_api_key(key)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# on-site service requests
# --------------------------------------------------------------------------- #
@router.get("/api/billing/service-requests")
def service_requests():
    return {"requests": billing.list_service_requests()}


@router.post("/api/billing/service-requests")
async def add_service_request(request: Request):
    b = await request.json()
    return billing.create_service_request(str(b.get("kind", "periodic_service")),
                                          str(b.get("note", "")), str(b.get("contact", "")))


@router.post("/api/billing/service-requests/{req_id}/status")
async def set_service_status(req_id: int, request: Request):
    b = await request.json()
    billing.set_service_request_status(req_id, str(b.get("status", "open")))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# data retention (the plan's headline limit)
# --------------------------------------------------------------------------- #
@router.get("/api/billing/retention")
def retention_preview():
    """DRY-RUN: how many rows the active plan's retention would prune. Never deletes."""
    return billing.enforce_retention(dry_run=True)


@router.post("/api/billing/retention/run")
def retention_run():
    """LIVE prune — only actually deletes if env BILLING_RETENTION_ENFORCE=1."""
    return billing.enforce_retention(dry_run=False)
