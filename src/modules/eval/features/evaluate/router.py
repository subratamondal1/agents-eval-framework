from __future__ import annotations
from fastapi import APIRouter
from .schemas import EvaluateRequest, EvaluateResponse
from .logic import execute_evaluation

router = APIRouter(prefix="/evaluate", tags=["evaluate"])

@router.post("/", status_code=200, response_model=EvaluateResponse)
async def evaluate_endpoint(
    data: EvaluateRequest,
) -> EvaluateResponse:
    return await execute_evaluation(data=data)
