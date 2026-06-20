from __future__ import annotations
from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    model: str = Field(
        default="accounts/fireworks/models/deepseek-v4-flash",
        description="Model to evaluate",
    )
    prompt: str = Field(description="Prompt for the evaluation")
    images: list[str] = Field(default_factory=list, description="List of image URLs")
    max_tokens: int = Field(default=131072, description="Max tokens")
    temperature: float = Field(default=0.1, description="Temperature")
    top_p: float = Field(default=1.0, description="Top P")
    top_k: int = Field(default=40, description="Top K")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")


class EvaluateResponse(BaseModel):
    model: str = Field(description="Model used")
    response_text: str = Field(description="Response from the model")
    prompt_tokens: int = Field(description="Tokens used in prompt")
    completion_tokens: int = Field(description="Tokens used in completion")
    total_tokens: int = Field(description="Total tokens used")
