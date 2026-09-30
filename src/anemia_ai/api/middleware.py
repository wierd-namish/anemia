"""
Custom middleware and global exception handlers.
"""

import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from anemia_ai.core.exceptions import AnemiaAIError
from anemia_ai.core.logging import logger


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Attaches a unique Request-ID to incoming requests and logs execution time."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
        t0 = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response


async def anemia_exception_handler(request: Request, exc: AnemiaAIError) -> JSONResponse:
    """Handles domain-specific Anemia AI exceptions."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("AnemiaAIError on %s: %s (req_id=%s)", request.url.path, exc.message, req_id)
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "request_id": req_id,
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches unhandled exceptions and prevents internal traceback leakage."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception("Unhandled server error on %s (req_id=%s): %s", request.url.path, req_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "An unexpected error occurred during processing.",
            "request_id": req_id,
        },
    )
