from __future__ import annotations
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/healthz", status_code=200)
async def liveness() -> dict[str, str]:
    return {"status": "ok"}
