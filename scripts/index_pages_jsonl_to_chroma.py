
"""
Индексирование page-level JSONL (pages.jsonl) в локальный Chroma.

Ожидаемый формат строк в JSONL:
  {
    "source_url": "...",
    "title": "...",
    "text": "...",
    "html_path": "..."
  }

Результат:
  persist_directory/collection_name заполнены чанками, каждый чанк имеет metadata:
    source_url, title, chunk_index, html_path
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config_loader import load_config


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stable_chunk_id(source_url: str | None, html_path: str, chunk_index: int, chunk_text: str) -> str:
    chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    base = f"{source_url or ''}|{html_path}|{chunk_index}|{chunk_hash}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="pages.jsonl -> Chroma (LangChain)")
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Путь к config.yaml")

    p.add_argument("--input", type=Path, default=None, help="Путь к pages.jsonl")
    p.add_argument("--persist-dir", type=Path, default=None, help="Chroma persist_directory")
    p.add_argument("--collection", type=str, default=None, help="Chroma collection_name")
    p.add_argument("--embedding-model", type=str, default=None, help="Embedding модель (sentence-transformers)")
    p.add_argument("--chunk-size", type=int, default=None, help="Размер чанка")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Перекрытие чанка")
    p.add_argument("--min-chunk-chars", type=int, default=None, help="Минимальная длина чанка (в символах)")
    p.add_argument("--limit", type=int, default=None, help="Опционально: обработать только N страниц (0 = без лимита)")
    p.add_argument("--recreate", action="store_true", help="Полностью пересоздать persist_directory")
    p.add_argument("--batch-size", type=int, default=None, help="Добавлять чанки в Chroma батчами")
    args = p.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or Path(cfg["paths"]["pages_jsonl_path"])
    persist_dir = args.persist_dir or Path(cfg["paths"]["chroma_persist_dir"])
    collection = args.collection or str(cfg["paths"]["chroma_collection"])
    embedding_model = args.embedding_model or str(cfg["embeddings"]["model_name"])

    chunk_size = int(args.chunk_size) if args.chunk_size is not None else int(cfg["chunking"]["chunk_size"])
    chunk_overlap = (
        int(args.chunk_overlap) if args.chunk_overlap is not None else int(cfg["chunking"]["chunk_overlap"])
    )
    min_chunk_chars = (
        int(args.min_chunk_chars) if args.min_chunk_chars is not None else int(cfg["chunking"]["min_chunk_chars"])
    )
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg["indexing"]["batch_size"])
    limit = int(args.limit) if args.limit is not None else int(cfg["indexing"].get("limit", 0))

    if not input_path.is_file():
        print(f"Input JSONL not found: {input_path}", file=sys.stderr)
        return 1

    if args.recreate and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    
    try:
        from bs4 import XMLParsedAsHTMLWarning  

        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except Exception:
        pass

    
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    batch_docs: List[Document] = []
    batch_ids: List[str] = []

    page_count = 0
    chunk_count = 0

    with tqdm(total=None, desc="Indexing pages", unit="page") as bar:
        for obj in _iter_jsonl(input_path):
            page_count += 1
            if limit and page_count > limit:
                break

            text = str(obj.get("text") or "").strip()
            source_url = obj.get("source_url") or None
            title = str(obj.get("title") or "")
            html_path = str(obj.get("html_path") or "")

            if not text:
                continue

            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk or len(chunk) < min_chunk_chars:
                    continue
                chunk_id = _stable_chunk_id(source_url, html_path, i, chunk)

                batch_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_url": source_url,
                            "title": title,
                            "chunk_index": i,
                            "html_path": html_path,
                        },
                    )
                )
                batch_ids.append(chunk_id)
                chunk_count += 1

                if len(batch_docs) >= batch_size:
                    vectorstore.add_documents(batch_docs, ids=batch_ids)
                    batch_docs.clear()
                    batch_ids.clear()

            bar.update(1)

    if batch_docs:
        vectorstore.add_documents(batch_docs, ids=batch_ids)

    
    vectorstore.persist()

    print(
        f"Done. pages={page_count}, chunks={chunk_count}, persist_dir={persist_dir}, collection={collection}, limit={limit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

