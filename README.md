# LLM Gateway

## Run
    cp .env.example .env   # add OPENAI_API_KEY
    docker compose up -d redis
    pip install -r requirements.txt
    uvicorn app.main:app --reload

## Try it
    curl localhost:8000/v1/chat/completions \
      -H "Authorization: Bearer gw-search-123" -H "Content-Type: application/json" \
      -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'

## Test
    pytest
