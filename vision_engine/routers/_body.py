"""4.0.425 — read a request's JSON body in an async dependency, so the endpoint itself
can be a plain `def` that FastAPI runs in its worker threadpool.

Before, ~40 endpoints were `async def` only because they did `await request.json()` —
and then ran ALL their blocking work (DB reads/writes, config save with fsync, Redis,
V4L2 ioctls, RTSP opens, HTTP calls to YOLO / the AI trainer / payment gateways) on
the single event loop, freezing every other request — /health included — while they ran.

Usage:
    @router.post("/x")
    def handler(request: Request, payload=Depends(json_body)):
        data = body(payload)          # exactly where `await request.json()` used to be

`body()` re-raises a parse error (empty / malformed body) at that same spot, so each
endpoint's own try/except sees the same exception it always did — behaviour unchanged.
"""
from fastapi import Request


class _BodyError:
    __slots__ = ("exc",)

    def __init__(self, exc: Exception):
        self.exc = exc


async def json_body(request: Request):
    try:
        return await request.json()
    except Exception as e:      # handed to the endpoint, re-raised by body()
        return _BodyError(e)


def body(payload):
    if isinstance(payload, _BodyError):
        raise payload.exc
    return payload
