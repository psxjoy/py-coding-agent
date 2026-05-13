from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Message:
    """Core message type for LLM converstaion"""
    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion"""
    text: str
    stop_reason: str
    usage: dict = field(default_factory=dict)


@dataclass
class ChatOptions:
    """Options for chat completion"""
    system: str | None = None
    max_tokens: int | None = None


class LLMProvider(Protocol):
    """Unified interface for LLM providers."""

    async def chat(
            self, messages: list[Message], options: ChatOptions | None = None
    ) -> ChatResponse: ...
