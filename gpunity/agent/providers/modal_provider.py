"""Modal-hosted vLLM provider for GPUnity.

Routes LLM inference to a vLLM endpoint running on Modal,
using the OpenAI-compatible API. Reuses OpenAIProvider's
message/tool conversion logic.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from gpunity.agent.providers.openai_provider import OpenAIProvider
from gpunity.utils.logging import get_logger

logger = get_logger("gpunity.agent.providers.modal")

DEFAULT_MODAL_MODEL = "Qwen/Qwen3-8B"


class ModalProvider(OpenAIProvider):
    """LLM provider that routes to a Modal-hosted vLLM endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODAL_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("MODAL_VLLM_URL", "")
        if not self.base_url:
            raise ValueError(
                "MODAL_VLLM_URL not set. Deploy modal_app.py first:\n"
                "  modal deploy modal_app.py\n"
                "Then set MODAL_VLLM_URL to the endpoint URL."
            )
        # vLLM doesn't require a real API key but the openai client needs one
        self.api_key = api_key or "modal-vllm-key"
        self.model = model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a completion request to Modal-hosted vLLM.

        Reuses OpenAIProvider's message/tool format conversion
        but points the client at the Modal vLLM endpoint.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        import json

        # Point client at Modal vLLM endpoint
        vllm_url = self.base_url.rstrip("/")
        if not vllm_url.endswith("/v1"):
            vllm_url = vllm_url + "/v1"

        client = openai.OpenAI(api_key=self.api_key, base_url=vllm_url)

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
        # Append /no_think to system prompt to suppress Qwen3 reasoning tags
        openai_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                sys_content = msg.get("content", "")
                if "/no_think" not in sys_content:
                    sys_content = sys_content + "\n/no_think"
                openai_messages.append({"role": "system", "content": sys_content})
            elif role == "user":
                content = msg.get("content", "")
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

        logger.info(f"Calling Modal vLLM endpoint: {vllm_url}")
        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        # Strip Qwen3 <think> reasoning tags from output
        import re
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
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
