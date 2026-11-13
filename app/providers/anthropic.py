"""Anthropic provider: translates OpenAI chat format <-> Anthropic Messages API."""
import httpx
from fastapi import HTTPException

from .base import ProviderError

STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                "tool_use": "tool_calls"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def to_anthropic(payload: dict, model: str) -> dict:
    system = "\n".join(_text(m["content"]) for m in payload["messages"] if m["role"] == "system")
    messages = [{"role": m["role"], "content": _text(m["content"])}
                for m in payload["messages"] if m["role"] in ("user", "assistant")]
    body = {"model": model, "messages": messages,
            "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens") or 1024}
    if system:
        body["system"] = system
    for k in ("temperature", "top_p"):
        if k in payload:
            body[k] = payload[k]
    stop = payload.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    return body


def to_openai(data: dict) -> dict:
    u = data.get("usage", {})
    return {
        "id": data["id"], "object": "chat.completion", "model": data["model"],
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": _text(data["content"])},
                     "finish_reason": STOP_REASONS.get(data.get("stop_reason"), "stop")}],
        "usage": {"prompt_tokens": u.get("input_tokens", 0),
                  "completion_tokens": u.get("output_tokens", 0),
                  "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0)},
    }


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, model: str,
                 base_url: str = "https://api.anthropic.com/v1"):
        self._client, self._api_key, self._model = client, api_key, model
        self._base_url = base_url.rstrip("/")

    async def chat(self, payload: dict) -> dict:
        try:
            resp = await self._client.post(
                f"{self._base_url}/messages", json=to_anthropic(payload, self._model),
                headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"})
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise ProviderError(self.name, None, repr(e)) from e
        if resp.status_code >= 500 or resp.status_code in (429, 529):
            raise ProviderError(self.name, resp.status_code, resp.text[:200])
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.json())
        return to_openai(resp.json())
