from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    # Anthropic (enabled when a key is set)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"
    # Ollama (enabled when a model is set)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = ""
    retry_max_attempts: int = 3
    breaker_failure_threshold: int = 5
    breaker_reset_timeout_s: float = 30.0
    redis_url: str = "redis://localhost:6379/0"
    upstream_timeout_s: float = 30.0
    # Day 1 stand-in for a Postgres api_keys table: "name:key:rpm,name:key:rpm"
    gateway_keys: str = ""
    # Usage tracking (disabled when empty) and admin endpoints (disabled when empty)
    database_url: str = ""
    admin_key: str = ""
    # Response cache: TTL in seconds (0 disables), max cached response size in bytes
    cache_ttl_s: int = 3600
    cache_max_bytes: int = 262144
    # Semantic cache (off unless EMBED_PROVIDER is "ollama" or "openai")
    embed_provider: str = ""
    embed_model: str = ""              # default: nomic-embed-text (ollama) / text-embedding-3-small
    embed_prefix: str = ""            # text prepended before embedding (nomic wants 'search_query: ')
    semantic_threshold: float = 0.92   # cosine similarity needed for a hit; tune with tools/eval_semantic.py
    semantic_max_entries: int = 500    # per bucket (api key + context)
    semantic_max_wording_overlap: float = 0.4  # minimal-pair guard; set 2 to disable
    key_cache_ttl_s: float = 10.0      # how long a looked-up key is trusted before re-checking Postgres
    max_body_bytes: int = 1_048_576    # request bodies larger than this get 413 (0 = unlimited)
    max_concurrent_per_key: int = 0    # requests served at once per key, per process (0 = unlimited)
    access_log: bool = True            # one JSON line per chat request, with the request id
    log_level: str = "INFO"
    metrics_token: str = ""           # if set, GET /metrics requires 'Authorization: Bearer <token>'
    embed_timeout_s: float = 2.0       # slower than this = skip the semantic lookup (fail open)


settings = Settings()
