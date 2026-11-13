"""Embedding client. Ollama and OpenAI both serve the OpenAI-style POST {base}/embeddings,
so one class covers both."""
import httpx


class Embedder:
    def __init__(self, client: httpx.AsyncClient, base_url: str, model: str, api_key: str = "",
                 prefix: str = ""):
        self._client, self._base, self._model, self._key = client, base_url.rstrip("/"), model, api_key
        self._prefix = prefix      # e.g. "search_query: " for nomic-embed-text

    async def embed(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        r = await self._client.post(f"{self._base}/embeddings", headers=headers,
                                    json={"model": self._model, "input": self._prefix + text})
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
