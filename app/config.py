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


settings = Settings()
