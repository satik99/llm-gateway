import httpx
from fastapi import HTTPException

from .base import ProviderError


class OpenAIProvider:
    """Any OpenAI-compatible endpoint: OpenAI itself, Ollama, vLLM, ..."""

    def __init__(self, client: httpx.AsyncClient, *, name: str = "openai",
                 base_url: str = "https://api.openai.com/v1", api_key: str = "",
                 model_override: str | None = None):
        self.name = name
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_override = model_override  # fallbacks use a different model name

    async def chat(self, payload: dict) -> dict:
        if self._model_override:
            payload = {**payload, "model": self._model_override}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise ProviderError(self.name, None, repr(e)) from e

        if resp.status_code >= 500 or resp.status_code == 429:
            raise ProviderError(self.name, resp.status_code, resp.text[:200])
        if resp.status_code >= 400:
            # Client error (bad request): retrying or falling back won't help.
            raise HTTPException(resp.status_code, resp.json())
        return resp.json()
