"""Measure semantic-cache quality and pick a threshold.

Each case is (question_a, question_b, same_meaning). We embed both with the REAL embedding model
and sweep the threshold. At threshold T a pair counts as a cache hit if cosine(a, b) >= T:
  hit rate on paraphrases  = share of same-meaning pairs served from cache   (want HIGH)
  false-hit rate           = share of different-meaning pairs wrongly served (want ~0)

Usage:  python tools/eval_semantic.py [--provider ollama|openai] [--model NAME] [--json out.json]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.semantic import wording_conflict  # noqa: E402

# (a, b, same_meaning)  -- same meaning: the SAME answer is correct for both.
# Different meaning includes deliberate near-misses: that is where false hits happen.
PAIRS = [
    # --- paraphrases (should hit) ---
    ("What is your refund policy?", "How do I get my money back?", True),
    ("How do I reset my password?", "I forgot my password, how can I change it?", True),
    ("How can I cancel my subscription?", "What are the steps to end my plan?", True),
    ("What payment methods do you accept?", "Which ways can I pay?", True),
    ("Where can I find my invoices?", "How do I download my past bills?", True),
    ("How do I contact support?", "What's the way to reach your help team?", True),
    ("What are your opening hours?", "When are you open?", True),
    ("Do you ship internationally?", "Can you deliver to other countries?", True),
    ("How long does shipping take?", "How many days until my order arrives?", True),
    ("How do I update my email address?", "I need to change the email on my account.", True),
    ("Is my data encrypted?", "Do you encrypt customer data?", True),
    ("How do I delete my account?", "I want to permanently close my account.", True),
    ("Do you offer a student discount?", "Is there a cheaper price for students?", True),
    ("What is the capital of France?", "Which city is the capital of France?", True),
    ("Explain what an API is.", "What does API mean and how does it work?", True),
    ("How do I enable two-factor authentication?", "Steps to turn on 2FA?", True),
    ("Can I change my plan later?", "Is it possible to switch plans after signing up?", True),
    ("What is the difference between TCP and UDP?", "How do TCP and UDP differ?", True),
    ("How do I export my data?", "Can I download all of my data?", True),
    ("Why was my card declined?", "My payment didn't go through, what happened?", True),
    ("How do I install Python on Windows?", "What's the way to set up Python on a Windows PC?", True),
    ("What is a REST API?", "Can you explain RESTful APIs?", True),
    ("How do I add a team member?", "Steps to invite a colleague to my workspace?", True),
    ("What is your uptime guarantee?", "How reliable is your service, is there an SLA?", True),
    ("Translate 'good morning' to Spanish.", "How do you say 'good morning' in Spanish?", True),
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?", True),
    ("What is machine learning?", "Explain machine learning in simple terms.", True),
    ("How do I get a receipt for my purchase?", "Where do I find the receipt for my order?", True),
    ("Do you have a mobile app?", "Is there an iOS or Android app?", True),
    ("How do I report a bug?", "Where can I submit a bug report?", True),
    # --- different meaning, clearly different (should miss) ---
    ("What is your refund policy?", "How do I reset my password?", False),
    ("How can I cancel my subscription?", "What payment methods do you accept?", False),
    ("What are your opening hours?", "Do you ship internationally?", False),
    ("What is the capital of France?", "What is the capital of Germany?", False),
    ("How do I install Python on Windows?", "How do I install Python on Linux?", False),
    ("Explain what an API is.", "Explain what a database is.", False),
    ("Translate 'good morning' to Spanish.", "Translate 'good morning' to French.", False),
    ("How do I reverse a list in Python?", "How do I sort a list in Python?", False),
    # --- near-misses: similar wording, DIFFERENT correct answer (the dangerous ones) ---
    ("How do I enable two-factor authentication?", "How do I disable two-factor authentication?", False),
    ("How do I add a team member?", "How do I remove a team member?", False),
    ("Can I upgrade my plan?", "Can I downgrade my plan?", False),
    ("How do I delete my account?", "How do I delete a single message?", False),
    ("What is your refund policy for subscriptions?", "What is your refund policy for hardware?", False),
    ("How long does shipping take within the US?", "How long does shipping take to Europe?", False),
    ("What is 15% of 200?", "What is 15% of 300?", False),
    ("Convert 10 km to miles.", "Convert 10 miles to km.", False),
    ("Is the Pro plan $20 per month?", "Is the Pro plan $20 per year?", False),
    ("Does the app work offline?", "Does the app work online only?", False),
    ("How do I enable dark mode?", "How do I disable dark mode?", False),
    ("What are the side effects of ibuprofen?", "What are the side effects of paracetamol?", False),
    ("Write a poem about the sea.", "Write a poem about the sky.", False),
    ("Summarize the plot of Hamlet.", "Summarize the plot of Macbeth.", False),
    ("How do I log in?", "How do I log out?", False),
    ("Who won the 2018 World Cup?", "Who won the 2014 World Cup?", False),
    ("What is the time complexity of quicksort?", "What is the time complexity of mergesort?", False),
    ("How do I increase my rate limit?", "How do I decrease my rate limit?", False),
    ("Can I share my account with someone?", "Can I transfer my account to someone?", False),
    ("What happens if I miss a payment?", "What happens if I pay early?", False),
    ("Where is your office in London?", "Where is your office in New York?", False),
    ("How do I turn on notifications?", "How do I turn off notifications?", False),
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=settings.embed_provider or "ollama")
    ap.add_argument("--model", default=settings.embed_model)
    ap.add_argument("--prefix", default=settings.embed_prefix)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    async with httpx.AsyncClient(timeout=30) as http:
        if args.provider == "ollama":
            emb = Embedder(http, settings.ollama_base_url, args.model or "nomic-embed-text", prefix=args.prefix)
        else:
            emb = Embedder(http, settings.openai_base_url, args.model or "text-embedding-3-small",
                           settings.openai_api_key, prefix=args.prefix)
        texts = sorted({t for a, b, _ in PAIRS for t in (a, b)})
        vecs = {}
        for t in texts:
            v = np.asarray(await emb.embed(t), dtype=np.float32)
            vecs[t] = v / np.linalg.norm(v)
    sims = np.array([float(vecs[a] @ vecs[b]) for a, b, _ in PAIRS])
    same = np.array([s for _, _, s in PAIRS])
    n_same, n_diff = int(same.sum()), int((~same).sum())
    print(f"model={args.provider}:{args.model or 'default'} prefix={args.prefix!r}  pairs: {n_same} same-meaning, {n_diff} different")
    print(f"similarity  same-meaning: mean {sims[same].mean():.3f} min {sims[same].min():.3f}")
    print(f"similarity  different   : mean {sims[~same].mean():.3f} max {sims[~same].max():.3f}\n")
    guard = np.array([wording_conflict(a, b, settings.semantic_max_wording_overlap) for a, b, _ in PAIRS])
    print(f"wording guard (overlap >= {settings.semantic_max_wording_overlap}) blocks "
          f"{int(guard[same].sum())}/{n_same} paraphrases and {int(guard[~same].sum())}/{n_diff} different pairs\n")
    print(f"{'thr':>5} | {'EMBEDDING ONLY':^29} | {'WITH WORDING GUARD':^29}")
    print(f"{'':>5} | {'hit rate':>13} {'false hits':>14} | {'hit rate':>13} {'false hits':>14}")
    rows = []
    for thr in np.arange(0.70, 1.0001, 0.02):
        hit = float((sims[same] >= thr).mean())
        false = float((sims[~same] >= thr).mean())
        ghit = float(((sims[same] >= thr) & ~guard[same]).mean())
        gfalse = float(((sims[~same] >= thr) & ~guard[~same]).mean())
        rows.append({"threshold": round(float(thr), 2), "hit_rate_embedding_only": hit,
                     "false_hit_rate_embedding_only": false, "hit_rate": ghit, "false_hit_rate": gfalse})
        print(f"{thr:5.2f} | {hit*100:12.1f}% {false*100:13.1f}% | {ghit*100:12.1f}% {gfalse*100:13.1f}%")
    ok = [r for r in rows if r["false_hit_rate"] == 0]
    best = max(ok, key=lambda r: r["hit_rate"]) if ok else None
    print()
    if best:
        print(f"Lowest-risk pick (with guard): threshold {best['threshold']} -> {best['hit_rate']*100:.1f}% of paraphrases "
              f"hit with 0 false hits on this set. Set SEMANTIC_THRESHOLD={best['threshold']} (add a small margin).")
    else:
        print("No threshold gave 0 false hits; pick the highest-threshold row you can accept.")
    print("\nWorst false-hit candidates (different meaning, highest similarity):")
    for i in np.argsort(-np.where(same, -1, sims))[:5]:
        print(f"  {sims[i]:.3f}  {PAIRS[i][0]!r}  vs  {PAIRS[i][1]!r}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "best": best}, indent=2))


asyncio.run(main())
