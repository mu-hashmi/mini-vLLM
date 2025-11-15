"""Pydantic request/response schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request schema for /generate endpoint."""

    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(
        default=100, ge=1, le=2048, description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling"
    )
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")


class TokenDelta(BaseModel):
    """Single token delta for streaming."""

    token: str
    text: str


class GenerateResponse(BaseModel):
    """Response schema for /generate endpoint."""

    text: str = Field(..., description="Generated text")
    tokens: List[str] = Field(..., description="Generated tokens")
    num_tokens: int = Field(..., description="Number of tokens generated")


class StreamChunk(BaseModel):
    """Single chunk in streaming response."""

    token: str = Field(..., description="Generated token")
    text: str = Field(..., description="Accumulated text so far")
    finished: bool = Field(default=False, description="Whether generation is complete")
    ttft: Optional[float] = Field(
        default=None, description="Time to first token in seconds"
    )
    tokens_per_sec: Optional[float] = Field(
        default=None, description="Tokens per second"
    )
