import httpx

from app.providers.anthropic import AnthropicProvider, to_anthropic


def test_request_translation():
    body = to_anthropic({"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"}], "temperature": 0.2, "stop": "X"}, "m")
    assert body["system"] == "Be brief."
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 1024 and body["stop_sequences"] == ["X"]


async def test_response_translation():
    def handler(req):
        assert req.headers["x-api-key"] == "k"
        return httpx.Response(200, json={
            "id": "msg_1", "model": "m", "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 2}})
    p = AnthropicProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                          api_key="k", model="m")
    r = await p.chat({"messages": [{"role": "user", "content": "hi"}]})
    assert r["choices"][0]["message"]["content"] == "hello"
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["total_tokens"] == 7
