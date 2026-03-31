
"""
Вход: каталог с офлайн-зеркалом Moodle Docs (или подкаталог), внутри которого лежат `.html`.
Выход: JSONL по исходным страницам (page-level):
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Iterable, Iterator


try:
    from tqdm import tqdm
except ImportError:  
    
    def tqdm(it: Iterable[Path], **_: object) -> Iterable[Path]:  
        return it


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
from extract_moodle_doc_text import extract_important_text  

from configs.config_loader import load_config  


def iter_html_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".html":
            yield input_path
        return
    yield from input_path.rglob("*.html")


def _jsonl_write(fp, obj: dict) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Парсинг офлайн Moodle Docs HTML -> JSONL page-level")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Путь к config.yaml (по умолчанию configs/config.yaml).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Каталог или один .html файл офлайн-версии Moodle Docs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Каталог под JSONL артефакты (создаётся автоматически).",
    )
    p.add_argument("--min-chars", type=int, default=None, help="Минимальный размер извлечённого текста.")
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Ограничение на размер текста страницы (сверх — skip, чтобы избежать аномалий).",
    )
    p.add_argument("--write-errors", action="store_true", help="Писать skipped/errors в errors.jsonl.")
    p.add_argument("--limit", type=int, default=None, help="Опционально: обработать только N файлов (0 = без лимита).")

    args = p.parse_args()

    cfg = load_config(args.config)

    input_path: Path = args.input or Path(cfg["paths"]["offline_docs_root"])
    out_dir: Path = args.output_dir or Path(cfg["paths"]["moodle_docs_output_dir"])
    min_chars: int = int(args.min_chars if args.min_chars is not None else cfg["parse_html"]["min_chars"])
    max_chars: int = int(args.max_chars if args.max_chars is not None else cfg["parse_html"]["max_chars"])
    limit: int = int(args.limit) if args.limit is not None else int(cfg["parse_html"]["limit"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_path = out_dir / "pages.jsonl"
    errors_path = out_dir / "errors.jsonl"

    total = 0
    kept = 0
    skipped = 0
    errors = 0

    with pages_path.open("w", encoding="utf-8") as pages_fp, (
        errors_path.open("w", encoding="utf-8") if args.write_errors else _NullWriter()
    ) as errors_fp:
        for path in tqdm(iter_html_files(input_path), desc="Parsing HTML", unit="file"):
            if limit and total >= limit:
                break

            total += 1
            try:
                html = path.read_text(encoding="utf-8", errors="replace")
                canonical_url, title, text = extract_important_text(html)

                reason = None
                if not canonical_url:
                    reason = "no_canonical"
                elif not text.strip():
                    reason = "no_mw_content"
                elif len(text) < min_chars:
                    reason = "too_short"
                elif len(text) > max_chars:
                    reason = "too_long"

                if reason is not None:
                    skipped += 1
                    if args.write_errors:
                        _jsonl_write(
                            errors_fp,
                            {"html_path": str(path), "reason": reason},
                        )
                    continue

                _jsonl_write(
                    pages_fp,
                    {
                        "source_url": canonical_url,
                        "title": title,
                        "text": text,
                        "html_path": str(path),
                    },
                )
                kept += 1
            except Exception as e:  
                errors += 1
                skipped += 1
                if args.write_errors:
                    _jsonl_write(
                        errors_fp,
                        {
                            "html_path": str(path),
                            "reason": "parse_error",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )

    manifest = {
        "input": str(input_path),
        "total": total,
        "kept": kept,
        "skipped": skipped,
        "errors": errors,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "limit": limit,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. total={total}, kept={kept}, skipped={skipped}, errors={errors}")
    return 0


class _NullWriter:
    """Контекст-менеджер-заглушка для ошибок, если `--write-errors` не включён."""

    def write(self, _: str) -> None:  
        return

    def flush(self) -> None:  
        return

    def close(self) -> None:  
        return

    def __enter__(self):  
        return self

    def __exit__(self, exc_type, exc, tb):  
        return False


if __name__ == "__main__":
    raise SystemExit(main())

