"""
Поиск по локальному Chroma (через LangChain).

Используется тот же embedding model и разбиение на чанки, что и при индексации.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config_loader import load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Chroma retrieval (LangChain)")
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Путь к config.yaml")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--persist-dir", type=Path, default=None)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--max-print-chars", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    persist_dir = args.persist_dir or Path(cfg["paths"]["chroma_persist_dir"])
    collection = args.collection or str(cfg["paths"]["chroma_collection"])
    embedding_model = args.embedding_model or str(cfg["embeddings"]["model_name"])
    k = int(args.k) if args.k is not None else int(cfg["retrieval"]["top_k"])
    max_print_chars = (
        int(args.max_print_chars) if args.max_print_chars is not None else int(cfg["retrieval"]["max_print_chars"])
    )

    if not persist_dir.exists():
        print(f"Chroma persist dir not found: {persist_dir}", file=sys.stderr)
        return 1

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    hits = vectorstore.similarity_search_with_score(args.query, k=k)

    
    
    for rank, (doc, score) in enumerate(hits, start=1):
        source_url = doc.metadata.get("source_url")
        title = doc.metadata.get("title")
        chunk_index = doc.metadata.get("chunk_index")
        snippet = doc.page_content.replace("\n", " ")
        if len(snippet) > max_print_chars:
            snippet = snippet[:max_print_chars] + "..."

        print(f"[{rank}] score={score}")
        print(f"  title={title}")
        print(f"  source_url={source_url}")
        print(f"  chunk_index={chunk_index}")
        print(f"  text={snippet}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

