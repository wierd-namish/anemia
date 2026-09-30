"""
FastAPI Application Factory.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from anemia_ai.api.middleware import (
    RequestTracingMiddleware,
    anemia_exception_handler,
    global_exception_handler,
)
from anemia_ai.api.routes.health import router as health_router
from anemia_ai.api.routes.model_info import router as model_info_router
from anemia_ai.api.routes.predict import router as predict_router
from anemia_ai.config.constants import ENSEMBLE_VERSION
from anemia_ai.config.settings import get_settings
from anemia_ai.core.exceptions import AnemiaAIError
from anemia_ai.version import __version__


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="AI Nail Anemia Assessment API",
        description="Production-grade Medical-AI service for evaluating anemia probability from fingernail photographs.",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Custom Middleware
    app.add_middleware(RequestTracingMiddleware)

    # 2. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Exception Handlers
    app.add_exception_handler(AnemiaAIError, anemia_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # 4. Versioned API Routes (/api/v1/...)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(model_info_router, prefix="/api/v1")
    app.include_router(predict_router, prefix="/api/v1")

    # 5. Top-level Routes for direct backward compatibility (/health, /predict, etc.)
    app.include_router(health_router)
    app.include_router(model_info_router)
    app.include_router(predict_router)

    # 6. Static files and frontend hosting
    frontend_dir = settings.frontend_dir
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/style.css", include_in_schema=False)
        def serve_css():
            f = frontend_dir / "style.css"
            if f.exists():
                return FileResponse(str(f), media_type="text/css")
            return {"error": "style.css not found"}

        @app.get("/app.js", include_in_schema=False)
        def serve_js():
            f = frontend_dir / "app.js"
            if f.exists():
                return FileResponse(str(f), media_type="application/javascript")
            return {"error": "app.js not found"}

        @app.get("/", include_in_schema=False)
        def serve_frontend_index():
            index_file = frontend_dir / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            return {"message": "Frontend index.html not found"}

    return app


# Default app instance
app = create_app()
