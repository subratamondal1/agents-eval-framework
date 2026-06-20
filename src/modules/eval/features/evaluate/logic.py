from __future__ import annotations
import litellm
import structlog
from core.config import get_settings
from .schemas import EvaluateRequest, EvaluateResponse

logger = structlog.get_logger()


async def execute_evaluation(
    *,
    data: EvaluateRequest,
) -> EvaluateResponse:
    settings = get_settings()

    content: list[dict[str, str | dict[str, str]]] = [
        {"type": "text", "text": data.prompt}
    ]
    for img_url in data.images:
        content.append({"type": "image_url", "image_url": {"url": img_url}})

    logger.info(
        "evaluation_started",
        model=data.model,
        image_count=len(data.images),
    )

    response = await litellm.acompletion(
        model=f"fireworks_ai/{data.model}",
        api_key=settings.fireworks_api_key,
        messages=[{"role": "user", "content": content}],
        max_tokens=data.max_tokens,
        temperature=data.temperature,
        top_p=data.top_p,
        presence_penalty=data.presence_penalty,
        frequency_penalty=data.frequency_penalty,
        timeout=settings.llm_timeout_seconds,
    )

    logger.info(
        "evaluation_completed",
        model=data.model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )

    return EvaluateResponse(
        model=data.model,
        response_text=response.choices[0].message.content or "",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
