"""Cheap token estimation used to charge the TPM bucket BEFORE calling the provider.
The real usage is known only after the response, so we reconcile afterwards."""

DEFAULT_COMPLETION_GUESS = 256


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def estimate_prompt_tokens(payload: dict) -> int:
    """~4 chars per token, plus a few tokens of per-message overhead."""
    chars = sum(len(_text_of(m.get("content"))) for m in payload.get("messages", []))
    return chars // 4 + 4 * len(payload.get("messages", []))


def estimate_tokens(payload: dict) -> int:
    """Prompt estimate plus the max completion the caller allows."""
    prompt = estimate_prompt_tokens(payload)
    completion = (
        payload.get("max_completion_tokens")
        or payload.get("max_tokens")
        or DEFAULT_COMPLETION_GUESS
    )
    return prompt + int(completion)
