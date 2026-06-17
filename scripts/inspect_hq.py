"""Manual inspection of one chunk: HQ prompt + generated questions, and one query rewrite.

Makes 1-2 real Azure calls (cached). Use before a full ablation run to eyeball the
prompts and outputs. Pick the chunk/query with --chunk-index / --query-index.
"""

import argparse
from pathlib import Path

from ezmed.ingestion.enrichment import _parse_questions, enriched_text
from ezmed.ingestion.plaba import chunk_all, load_plaba
from ezmed.llm.client import LLMClient
from ezmed.llm.prompts import load_prompt
from ezmed.logging import configure_logging
from ezmed.settings import settings

PLABA_PATH = Path("data/plaba/data.json")
CHAT_CACHE_DIR = Path("data/processed/completions_cache")
RULE = "=" * 80


def main() -> None:
    args = _parse_args()
    configure_logging()
    if not settings.azure_openai_key or not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set")

    articles, queries = load_plaba(PLABA_PATH)
    chunks = chunk_all(articles)
    chunk = chunks[args.chunk_index]

    chat = LLMClient(
        deployment=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_key,
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        cache_dir=CHAT_CACHE_DIR,
    )

    # ---- HQ generation for one chunk ----
    hq_prompt = load_prompt("hq_generation")
    hq_user = hq_prompt.render_user(chunk_text=chunk.content, k=settings.hq_per_chunk)

    print(f"\n{RULE}\nCHUNK [{args.chunk_index}]  id={chunk.chunk_id}  pmid={chunk.pmid}"
          f"  section={chunk.section}  len={len(chunk.content)}\n{RULE}")
    print(chunk.content)

    print(f"\n{RULE}\nHQ SYSTEM PROMPT\n{RULE}\n{hq_prompt.system}")
    print(f"\n{RULE}\nHQ USER PROMPT (rendered, k={settings.hq_per_chunk})\n{RULE}\n{hq_user}")

    hq_resp = chat.complete(hq_prompt.system, hq_user)
    questions = _parse_questions(hq_resp.text, settings.hq_per_chunk)

    print(f"\n{RULE}\nRAW MODEL OUTPUT  (deployment={settings.azure_openai_chat_deployment}, "
          f"tokens={hq_resp.total_tokens})\n{RULE}\n{hq_resp.text}")
    print(f"\n{RULE}\nPARSED HQ QUESTIONS ({len(questions)})\n{RULE}")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print(f"\n{RULE}\nENRICHED TEXT THAT GETS EMBEDDED (content + HQ, Variant A)\n{RULE}")
    enriched_chunk = chunk.model_copy(update={"hq_questions": questions})
    print(enriched_text(enriched_chunk))

    # ---- Query rewriting for one query ----
    query = queries[args.query_index]
    qr_prompt = load_prompt("query_rewriting")
    qr_user = qr_prompt.render_user(lay_query=query["question"])
    qr_resp = chat.complete(qr_prompt.system, qr_user)

    print(f"\n{RULE}\nQUERY REWRITE  [qid={query['qid']}]\n{RULE}")
    print(f"  original : {query['question']}")
    print(f"  rewritten: {qr_resp.text.strip()}")
    print(f"  (tokens={qr_resp.total_tokens})\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
