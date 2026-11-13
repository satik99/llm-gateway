"""Request IDs and body-size limit (pure ASGI, so they also cover streaming and error paths)."""
import json
import logging
import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_RID_OK = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class RequestIdFilter(logging.Filter):
    """Puts the current request's id on every log record (use %(request_id)s in the format)."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class RequestIdMiddleware:
    """Every response carries X-Request-ID. A client-supplied one is reused if it looks sane
    (so a caller can correlate its own logs); otherwise we generate one."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _RID_OK.match(incoming) else uuid.uuid4().hex[:16]
        scope["request_id"] = rid
        token = request_id_var.set(rid)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message["headers"], (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


class _TooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Reject request bodies over `max_bytes` with 413, before they are parsed or forwarded.
    Checks Content-Length up front and also counts bytes for chunked uploads."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.max_bytes <= 0:
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)

        total, started, exceeded = 0, False, False

        async def counting_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    exceeded = True
                    raise _TooLarge()
            return message

        async def tracking_send(message):
            # If the limit was hit, whatever the framework turns that into (400/500) is replaced
            # by one clean 413, and anything else the app tries to send is dropped.
            nonlocal started
            if exceeded:
                if not started:
                    started = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _TooLarge:
            if not started:
                await self._reject(send)

    async def _reject(self, send):
        body = json.dumps({"error": f"request body larger than {self.max_bytes} bytes"}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})
