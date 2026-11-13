"""Per-model price table (USD per 1M tokens: input, output).

VERIFY these against the provider's pricing page before trusting the numbers;
they change. Unknown models return None (cost unknown) instead of a guess.
Local models (ollama) cost nothing in API spend.
"""

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

FREE_PROVIDERS = {"ollama"}


def estimate_cost(
    provider: str, model: str | None, prompt_tokens: int, completion_tokens: int
) -> float | None:
    if provider in FREE_PROVIDERS:
        return 0.0
    if not model:
        return None
    # Longest prefix wins, so "gpt-4o-mini-2024-07-18" matches "gpt-4o-mini", not "gpt-4o".
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            inp, out = PRICES[name]
            return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
    return None
