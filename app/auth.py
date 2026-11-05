from dataclasses import dataclass

from fastapi import Header, HTTPException

from .config import settings


@dataclass(frozen=True)
class ApiKey:
    name: str          # team / app name, used for cost attribution later
    key: str
    rpm: int           # requests per minute


def _load_keys() -> dict[str, ApiKey]:
    keys = {}
    for item in filter(None, (s.strip() for s in settings.gateway_keys.split(","))):
        name, key, rpm = item.split(":")
        keys[key] = ApiKey(name=name, key=key, rpm=int(rpm))
    return keys


def authenticate(authorization: str = Header(default="")) -> ApiKey:
    token = authorization.removeprefix("Bearer ").strip()
    api_key = _load_keys().get(token)
    if api_key is None:
        raise HTTPException(401, "invalid API key")
    return api_key
