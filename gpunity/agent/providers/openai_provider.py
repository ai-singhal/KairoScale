"""OpenAI GPT LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.agent.providers.openai")


class OpenAIProvider:
    """LLM provider using the OpenAI API.

    Implements the LLMProvider protocol for OpenAI models with
    function calling support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Provide via api_key param or env var."
            )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a completion request to OpenAI.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (will be converted to OpenAI function format).
            temperature: Sampling temperature.

        Returns:
            Dict with 'content' and optional 'tool_calls'.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        client = openai.OpenAI(api_key=self.api_key)

        # Convert Anthropic-style tools to OpenAI function format
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                })

        # Convert messages - merge system into messages format
        openai_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                openai_messages.append({"role": "system", "content": msg.get("content", "")})
            elif role == "user":
                content = msg.get("content", "")
                # Agent loop may pass Anthropic-style tool_result payloads as user content.
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and first.get("type") == "tool_result":
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": first.get("tool_use_id", ""),
                            "content": first.get("content", ""),
                        })
                        continue
                openai_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                }
                raw_tool_calls = msg.get("tool_calls")
                if isinstance(raw_tool_calls, list) and raw_tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {})),
                            },
                        }
                        for tc in raw_tool_calls
                    ]
                openai_messages.append(assistant_msg)
            elif role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_use_id", ""),
                    "content": msg.get("content", ""),
                })

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = None

        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": choice.finish_reason,
        }
