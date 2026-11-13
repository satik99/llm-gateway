from typing import Protocol


class ProviderError(Exception):
    """Upstream failed in a way that may be worth retrying / falling back on."""

    def __init__(self, provider: str, status: int | None, detail: str = ""):
        super().__init__(f"{provider} failed ({status}): {detail}")
        self.provider = provider
        self.status = status


class Provider(Protocol):
    name: str

    async def chat(self, payload: dict) -> dict: ...
