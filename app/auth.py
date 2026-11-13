from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from .config import settings


@dataclass(frozen=True)
class ApiKey:
    name: str          # team / app name, used for cost attribution later
    key: str
    rpm: int           # requests per minute
    tpm: int = 0       # tokens per minute (0 = unlimited)
    monthly_budget_usd: float = 0.0  # 0 = unlimited


def _load_keys() -> dict[str, ApiKey]:
    keys = {}
    for item in filter(None, (s.strip() for s in settings.gateway_keys.split(","))):
        # name:key:rpm[:tpm[:monthly_budget_usd]]
        parts = item.split(":")
        name, key, rpm = parts[0], parts[1], int(parts[2])
        tpm = int(parts[3]) if len(parts) > 3 and parts[3] else 0
        budget = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
        keys[key] = ApiKey(name=name, key=key, rpm=rpm, tpm=tpm, monthly_budget_usd=budget)
    return keys


async def authenticate(request: Request, authorization: str = Header(default="")) -> ApiKey:
    token = authorization.removeprefix("Bearer ").strip()
    api_key = _load_keys().get(token)                    # static keys from .env: no I/O
    if api_key is None and token:
        ks = getattr(request.app.state, "keystore", None)
        if ks is not None:
            try:
                api_key = await ks.lookup(token)
            except Exception:  # noqa: BLE001  (KeyStoreUnavailable)
                raise HTTPException(503, "key store unavailable, retry shortly")
    if api_key is None:
        raise HTTPException(401, "invalid API key")
    request.scope["gateway_key"] = api_key.name      # read by the metrics middleware
    return api_key
