"""
Structured logging, request correlation and cost events (shared CapsAI shape).

Every log line is one JSON object on stdout that Cloud Logging parses into
jsonPayload, with its special fields (severity, message, trace, spanId,
labels, sourceLocation). The same module is copied into every Python service;
keep the copies identical so log-based metrics work across services.

    import obs
    obs.setup_logging("audio-extractor")      # once, at import time of the app
    app.add_middleware(obs.RequestContextMiddleware)

    obs.log_event("gcs_download_done", bucket=b, object=o, bytes=n, duration_ms=ms)
    with obs.timed("ffmpeg_done", step="extract_audio") as f:
        ...; f["output_bytes"] = size
    httpx.post(url, headers={**obs.outbound_headers(), **auth})

Fields:
  severity, message, time, logger, service, event, request_id, job_id,
  project_id, logging.googleapis.com/trace (projects/<p>/traces/<id>).
Never pass secrets, signed URLs (use redact_url), emails or phone numbers.
Stdlib only (Python 3.9+).
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import unquote, urlsplit, urlunsplit

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID") or "new-backend-475414"
SERVICE = os.getenv("K_SERVICE") or "unknown"
REVISION = os.getenv("K_REVISION") or ""

REQUEST_ID_HEADER = "X-Request-Id"
CLOUD_TRACE_HEADER = "X-Cloud-Trace-Context"
TRACEPARENT_HEADER = "traceparent"
# Pub/Sub message attribute / Cloud Tasks header carrying the request id.
PUBSUB_REQUEST_ID_ATTR = "request_id"

_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

# Per-request / per-job context. The value is a mutable dict owned by one
# request (or job), so bind() from a threadpool worker that copied the
# context is still visible to the middleware's access-log line.
_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("obs_ctx", default=None)

_CONTEXT_KEYS = ("request_id", "job_id", "project_id", "user_id")
_INTERNAL_KEYS = ("trace", "span_id", "trace_sampled", "trace_header", "traceparent")

# ---------------------------------------------------------------- redaction

_URL_RE = re.compile(r"\b(https?|gs|s3)://[^\s\"'<>]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w/])\+\d[\d\s-]{8,15}\d\b")
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_BOT_TOKEN_RE = re.compile(r"\bbot\d{6,}:[A-Za-z0-9_-]{20,}")  # Telegram bot token in a URL path
_SECRET_KV_RE = re.compile(
    r"(?i)\b(token|secret|password|passwd|api[_-]?key|signature|x-goog-signature|authorization|cookie)"
    r"(\"?\s*[:=]\s*\"?)([^\s\"',&]{6,})"
)


def redact_url(url: Any) -> str:
    """URL without query string, fragment or userinfo (signed URLs carry credentials there)."""
    if url is None:
        return ""
    text = str(url)
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.split("?", 1)[0].split("#", 1)[0]
    if not parts.scheme:
        return text.split("?", 1)[0].split("#", 1)[0]
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def scrub(text: str) -> str:
    """Best-effort removal of URL query strings, emails, phones, bearer tokens and key=secret pairs."""
    if not text:
        return text
    if "://" in text:
        text = _URL_RE.sub(lambda m: redact_url(m.group(0)), text)
    if "@" in text:
        text = _EMAIL_RE.sub("[email]", text)
    if "+" in text:
        text = _PHONE_RE.sub("[phone]", text)
    text = _BEARER_RE.sub(lambda m: m.group(1) + " [redacted]", text)
    if ":" in text:
        text = _BOT_TOKEN_RE.sub("bot[redacted]", text)
    text = _SECRET_KV_RE.sub(lambda m: m.group(1) + m.group(2) + "[redacted]", text)
    return text


def parse_gcs_url(url: str) -> Tuple[str, str]:
    """(bucket, object) for gs://, storage.googleapis.com and <bucket>.storage.googleapis.com URLs; ("", "") otherwise."""
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return "", ""
    path = unquote(parts.path or "").lstrip("/")
    host = (parts.netloc or "").lower()
    if parts.scheme == "gs":
        return host, path
    if host in ("storage.googleapis.com", "storage.cloud.google.com"):
        if path.startswith("download/storage/v1/b/") or path.startswith("storage/v1/b/"):
            seg = path.split("/b/", 1)[1].split("/o/", 1)
            return seg[0], seg[1] if len(seg) > 1 else ""
        bucket, _, obj = path.partition("/")
        return bucket, obj
    if host.endswith(".storage.googleapis.com"):
        return host[: -len(".storage.googleapis.com")], path
    if host == "firebasestorage.googleapis.com" and "/b/" in parts.path:
        seg = path.split("b/", 1)[1].split("/o/", 1)
        return seg[0], seg[1] if len(seg) > 1 else ""
    return "", ""


# ---------------------------------------------------------------- region

_REGION: Optional[str] = None


def region() -> str:
    """Caller region: REGION / CLOUD_RUN_REGION / GCP_REGION env, else the metadata server (cached), else 'unknown'."""
    global _REGION
    if _REGION is not None:
        return _REGION
    value = os.getenv("REGION") or os.getenv("CLOUD_RUN_REGION") or os.getenv("GCP_REGION") or ""
    if not value and os.getenv("K_SERVICE"):
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/region",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=0.5) as resp:  # noqa: S310 - fixed internal URL
                value = resp.read().decode().rsplit("/", 1)[-1]
        except Exception:
            value = ""
    _REGION = value or "unknown"
    return _REGION


# ---------------------------------------------------------------- context


def get_context() -> Dict[str, Any]:
    ctx = _CTX.get()
    return ctx if ctx is not None else {}


def bind(**fields: Any) -> None:
    """Attach fields (job_id, project_id, request_id, ...) to every later log line in this request/job."""
    ctx = _CTX.get()
    if ctx is None:
        ctx = {}
        _CTX.set(ctx)
    for key, value in fields.items():
        if value is not None and value != "":
            ctx[key] = value


def request_id() -> str:
    return str(get_context().get("request_id") or "")


@contextlib.contextmanager
def context(**fields: Any) -> Iterator[Dict[str, Any]]:
    """Fresh context for a job/thread/worker loop iteration. Inherits the current trace + request id."""
    parent = get_context()
    ctx = {k: parent[k] for k in ("request_id",) + _INTERNAL_KEYS if k in parent}
    ctx.update({k: v for k, v in fields.items() if v is not None and v != ""})
    ctx.setdefault("request_id", new_request_id())
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def new_request_id() -> str:
    return uuid.uuid4().hex


def run_in_context(fn):
    """Wrap a callable so it runs with a copy of the current context (for threading.Thread / executors)."""
    ctx = contextvars.copy_context()

    def _runner(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return _runner


_RID_OK = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _clean_request_id(value: Optional[str]) -> str:
    value = (value or "").strip()
    return value if _RID_OK.match(value) else ""


def _parse_trace(cloud_trace: str, traceparent: str) -> Tuple[str, str, Optional[bool]]:
    """(trace_id, span_id, sampled) from traceparent or X-Cloud-Trace-Context."""
    if traceparent:
        parts = traceparent.strip().split("-")
        if len(parts) >= 4 and len(parts[1]) == 32:
            try:
                sampled = bool(int(parts[3][:2], 16) & 1)
            except ValueError:
                sampled = None
            return parts[1], parts[2], sampled
    if cloud_trace:
        head, _, opts = cloud_trace.strip().partition(";")
        trace_id, _, span = head.partition("/")
        if trace_id:
            sampled = None
            if "o=" in opts:
                sampled = opts.split("o=", 1)[1][:1] == "1"
            # X-Cloud-Trace-Context span ids are decimal; Cloud Logging wants 16 hex chars.
            try:
                span_hex = format(int(span), "016x") if span else ""
            except ValueError:
                span_hex = ""
            return trace_id, span_hex, sampled
    return "", "", None


def start_request_context(
    headers: Dict[str, str], request_id_value: Optional[str] = None
) -> Tuple[Dict[str, Any], contextvars.Token]:
    """Set the context for an incoming request (lower-cased header dict). Returns (ctx, token) for reset."""
    rid = _clean_request_id(request_id_value or headers.get("x-request-id")) or new_request_id()
    cloud_trace = headers.get("x-cloud-trace-context", "")
    traceparent = headers.get("traceparent", "")
    trace_id, span_id, sampled = _parse_trace(cloud_trace, traceparent)
    ctx: Dict[str, Any] = {"request_id": rid}
    if trace_id:
        ctx["trace"] = f"projects/{PROJECT_ID}/traces/{trace_id}"
    if span_id:
        ctx["span_id"] = span_id
    if sampled is not None:
        ctx["trace_sampled"] = sampled
    if cloud_trace:
        ctx["trace_header"] = cloud_trace
    if traceparent:
        ctx["traceparent"] = traceparent
    return ctx, _CTX.set(ctx)


def outbound_headers() -> Dict[str, str]:
    """Correlation headers to forward on calls to our other services."""
    ctx = get_context()
    out: Dict[str, str] = {}
    if ctx.get("request_id"):
        out[REQUEST_ID_HEADER] = str(ctx["request_id"])
    if ctx.get("trace_header"):
        out[CLOUD_TRACE_HEADER] = str(ctx["trace_header"])
    if ctx.get("traceparent"):
        out[TRACEPARENT_HEADER] = str(ctx["traceparent"])
    return out


def pubsub_attributes() -> Dict[str, str]:
    """Pub/Sub message attributes carrying the correlation id."""
    rid = request_id()
    return {PUBSUB_REQUEST_ID_ATTR: rid} if rid else {}


# ---------------------------------------------------------------- formatter

_STD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None)).keys()) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


class JsonFormatter(logging.Formatter):
    """One Cloud Logging-compatible JSON object per record."""

    def __init__(self, service: Optional[str] = None):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:  # bad %-args must never kill logging
            message = f"{record.msg} {record.args}"
        service = self.service or SERVICE
        payload: Dict[str, Any] = {
            "severity": _SEVERITY.get(record.levelno, record.levelname),
            "message": "",
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "logger": record.name,
            "service": service,
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        ctx = get_context()
        for key in _CONTEXT_KEYS:
            if ctx.get(key):
                payload[key] = ctx[key]
        if ctx.get("trace"):
            payload["logging.googleapis.com/trace"] = ctx["trace"]
            if ctx.get("span_id"):
                payload["logging.googleapis.com/spanId"] = ctx["span_id"]
            if ctx.get("trace_sampled") is not None:
                payload["logging.googleapis.com/trace_sampled"] = ctx["trace_sampled"]

        fields = getattr(record, "obs_fields", None) or {}
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and key not in ("event", "obs_fields") and not key.startswith("_"):
                payload.setdefault(key, _jsonable(scrub(value) if isinstance(value, str) else value))
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = scrub(value)
            payload[key] = _jsonable(value)

        if record.exc_info and record.exc_info[0] is not None:
            # Full traceback in `stack_trace` (Error Reporting groups on it); the
            # exception type + text is appended to `message` for quick reading.
            exc_type, exc_value = record.exc_info[0], record.exc_info[1]
            payload["stack_trace"] = scrub(self.formatException(record.exc_info))
            summary = f"{exc_type.__name__}: {exc_value}"
            message = f"{message}: {summary}" if message else summary
        elif record.stack_info:
            payload["stack_trace"] = scrub(record.stack_info)
        payload["message"] = scrub(message)

        labels = {"service": service}
        if event:
            labels["event"] = str(event)
        payload["logging.googleapis.com/labels"] = labels
        if record.levelno >= logging.WARNING and not event:
            payload["logging.googleapis.com/sourceLocation"] = {
                "file": record.pathname,
                "line": str(record.lineno),
                "function": record.funcName,
            }
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"severity": payload["severity"], "message": scrub(message), "service": service})


class _DropFilter(logging.Filter):
    def __init__(self, prefixes: Tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.msg if isinstance(record.msg, str) else ""
        return not msg.startswith(self.prefixes)


_configured = False


def setup_logging(service: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Route all stdlib logging (incl. uvicorn) through the JSON formatter on stdout.
    Level from LOG_LEVEL (default INFO). uvicorn's own access log is disabled:
    RequestContextMiddleware writes one `http_request` line per request instead.
    """
    global _configured, SERVICE
    if service and not os.getenv("K_SERVICE"):
        SERVICE = service
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    level_value = logging.getLevelName(level_name)
    if not isinstance(level_value, int):
        level_value = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level_value)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.asgi", "gunicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # The middleware logs unhandled exceptions with request context; drop uvicorn's duplicate.
    logging.getLogger("uvicorn.error").addFilter(_DropFilter(("Exception in ASGI application",)))
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True
    # Chatty third-party loggers stay at WARNING unless LOG_LEVEL=DEBUG.
    noisy_level = logging.DEBUG if level_value <= logging.DEBUG else logging.WARNING
    for name in ("urllib3", "google", "google_genai", "httpx", "httpcore", "botocore", "boto3", "s3transfer",
                 "multipart", "asyncio", "hpack", "grpc", "PIL", "matplotlib", "numba"):
        logging.getLogger(name).setLevel(max(noisy_level, level_value))

    if not _configured:
        _install_excepthooks()
    _configured = True
    return logging.getLogger(SERVICE)


def _install_excepthooks() -> None:
    log = logging.getLogger("obs")

    def _sys_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.error("Uncaught exception", exc_info=(exc_type, exc, tb), extra={"event": "unhandled_exception"})

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        log.error(
            "Uncaught exception in thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            extra={"event": "unhandled_exception"},
        )

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


# ---------------------------------------------------------------- events

_event_log = logging.getLogger("obs.event")


def log_event(event: str, level: int = logging.INFO, message: Optional[str] = None, **fields: Any) -> None:
    """One structured line with a stable snake_case `event` name and numeric fields."""
    if not _event_log.isEnabledFor(level):
        return
    _event_log.log(level, message or event, extra={"event": event, "obs_fields": fields})


def ms_since(start: float) -> int:
    """Milliseconds since a time.perf_counter() value."""
    return int((time.perf_counter() - start) * 1000)


@contextlib.contextmanager
def timed(event: str, level: int = logging.INFO, **fields: Any) -> Iterator[Dict[str, Any]]:
    """
    Log `event` once when the block exits, with duration_ms and status=ok|error
    (+ error=<ExceptionClass>). Add fields to the yielded dict inside the block.
    """
    data: Dict[str, Any] = dict(fields)
    start = time.perf_counter()
    try:
        yield data
    except BaseException as exc:
        data.setdefault("status", "error")
        data.setdefault("error", type(exc).__name__)
        raise
    finally:
        data["duration_ms"] = ms_since(start)
        data.setdefault("status", "ok")
        lvl = level if data.get("status") in ("ok", 200) else max(level, logging.WARNING)
        log_event(event, lvl, **data)


def storage_event(
    provider: str,
    op: str,
    bucket: str,
    object_name: str,
    nbytes: Optional[int],
    duration_ms: int,
    range_read: bool = False,
    status: str = "ok",
    level: int = logging.INFO,
    **extra: Any,
) -> None:
    """
    `gcs_download_done` / `gcs_upload_done` / `s3_download_done` / `s3_upload_done`
    (provider gcs|s3, op download|upload|copy|delete). Always use this for GCS/S3 I/O.
    A status other than "ok" is logged at WARNING or above.
    """
    if status != "ok":
        level = max(level, logging.WARNING)
    log_event(
        f"{provider}_{op}_done",
        level,
        provider=provider,
        op=op,
        bucket=bucket or "",
        object=redact_url(object_name or ""),
        bytes=int(nbytes) if nbytes is not None else None,
        duration_ms=int(duration_ms),
        range=bool(range_read),
        status=status,
        region=region(),
        **extra,
    )


# ---------------------------------------------------------------- ASGI middleware


class RequestContextMiddleware:
    """
    Pure ASGI middleware: request id + trace context in a contextvar, X-Request-Id
    echoed on responses, one `http_request` access line (route template, not raw
    path) per request except health checks, unhandled exceptions logged at ERROR
    with the stack trace.
    """

    def __init__(
        self,
        app,
        skip_paths=("/health", "/healthz", "/_ah/health", "/readyz", "/livez"),
        debug_routes=(),
    ):
        """
        skip_paths: raw paths never access-logged (unless 5xx).
        debug_routes: route templates (e.g. "/downloads/{job_id}" polled by clients)
        whose successful access lines are DEBUG instead of INFO.
        """
        self.app = app
        self.skip_paths = set(skip_paths)
        self.debug_routes = set(debug_routes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        ctx, token = start_request_context(headers)
        state = {"status": 500}
        start = time.perf_counter()

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                state["status"] = message.get("status", 500)
                hdrs = [(k, v) for k, v in (message.get("headers") or []) if k.lower() != b"x-request-id"]
                # ctx["request_id"] may have been re-bound by the handler (e.g. from Pub/Sub attributes).
                hdrs.append((b"x-request-id", str(ctx.get("request_id") or "").encode("latin-1", "replace")))
                message["headers"] = hdrs
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            state["status"] = 500
            logging.getLogger("obs").error(
                "Unhandled exception in %s %s",
                scope.get("method"),
                _route_template(scope),
                exc_info=True,
                extra={"event": "unhandled_exception"},
            )
            raise
        finally:
            status = int(state["status"])
            skip = (scope.get("path") or "") in self.skip_paths or scope.get("method") == "OPTIONS"
            if not skip or status >= 500:
                route = _route_template(scope)
                if status >= 500:
                    level = logging.WARNING
                elif route in self.debug_routes and status < 400:
                    level = logging.DEBUG
                else:
                    level = logging.INFO
                log_event(
                    "http_request",
                    level,
                    method=scope.get("method"),
                    route=route,
                    status=status,
                    latency_ms=ms_since(start),
                )
            _CTX.reset(token)


def _route_template(scope) -> str:
    """Route template (/api/projects/{project_id}), never the raw path with ids."""
    route = scope.get("route")
    path = getattr(route, "path_format", None) or getattr(route, "path", None)
    if path:
        return str(path)
    # Older FastAPI/Starlette don't put the matched route in scope: match it here.
    app = scope.get("app")
    try:
        from starlette.routing import Match

        for candidate in getattr(app, "routes", None) or []:
            match, _ = candidate.matches(scope)
            if match == Match.FULL:
                return str(getattr(candidate, "path_format", None) or getattr(candidate, "path", "") or "<unmatched>")
    except Exception:
        pass
    return "<unmatched>"
