from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.middleware import TimingMiddleware, RequestIdMiddleware
from modules.eval.public_api import evaluate_router
from runtime_api import router as health_router

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging()

    logger.info(
        "app_started",
        app_name=settings.app_name,
        env=settings.app_env,
    )

    yield

    logger.info("app_shutdown")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)

app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIdMiddleware)

app.include_router(health_router)
app.include_router(evaluate_router, prefix=settings.api_prefix)
