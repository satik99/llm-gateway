import pytest

from app.pricing import estimate_cost


def test_dated_model_matches_longest_prefix():
    # gpt-4o-mini must not be priced as gpt-4o
    c = estimate_cost("openai", "gpt-4o-mini-2024-07-18", 1_000_000, 1_000_000)
    assert c == pytest.approx(0.15 + 0.60)


def test_small_request():
    assert estimate_cost("openai", "gpt-4o-mini", 11, 7) == pytest.approx(
        (11 * 0.15 + 7 * 0.60) / 1_000_000
    )


def test_ollama_is_free():
    assert estimate_cost("ollama", "llama3.2", 500, 500) == 0.0


def test_unknown_model_is_none_not_zero():
    assert estimate_cost("openai", "some-new-model", 10, 10) is None
