"""Base LLM provider protocol."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by the optimization agent.

    All providers must implement the async complete() method, which
    takes messages and optional tool definitions and returns a response
    dict with 'content' and optional 'tool_calls'.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            temperature: Sampling temperature.

        Returns:
            Dict with keys:
            - 'content': str (text response)
            - 'tool_calls': list[dict] | None (tool use requests)
              Each tool_call: {'id': str, 'name': str, 'input': dict}
        """
        ...
