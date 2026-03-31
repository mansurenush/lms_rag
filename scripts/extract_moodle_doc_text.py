
"""
Извлечь из одной офлайн-страницы MoodleDocs (MediaWiki + HTTrack)
канонический URL, заголовок и основной текст статьи (без навигации, TOC, скриптов).

Пример:
  python scripts/extract_moodle_doc_text.py ./0a2f9817397e6cb2610a03c5bebd2b58.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Comment


def _parse_rlconf_title(html: str) -> str | None:
    m = re.search(r'"wgTitle"\s*:\s*"((?:[^"\\]|\\.)*)"', html)
    if not m:
        return None
    raw = m.group(1).replace("\\/", "/").replace('\\"', '"').replace("\\\\", "\\")
    return raw or None


def extract_important_text(html: str) -> tuple[str | None, str, str]:
    """
    Возвращает (canonical_url, page_title, article_plain_text).
    """
    parser = "lxml" if _have_lxml() else "html.parser"
    soup = BeautifulSoup(html, parser)

    canonical = None
    link = soup.find("link", rel=lambda x: x and "canonical" in x.split())
    if link and link.get("href"):
        canonical = link["href"].strip()

    title_tag = soup.find("title")
    page_title = ""
    if title_tag and title_tag.string:
        page_title = title_tag.string.strip()
        page_title = re.sub(r"\s*-\s*MoodleDocs\s*$", "", page_title, flags=re.I).strip()
    if not page_title:
        page_title = _parse_rlconf_title(html) or ""

    root = soup.select_one("#mw-content-text")
    if not root:
        return canonical, page_title, ""

    main = BeautifulSoup(str(root), parser)
    root2 = main.select_one("#mw-content-text")
    if root2 is None:
        root2 = main

    for tag in root2.find_all(["script", "style", "noscript"]):
        tag.decompose()

    for c in root2.find_all(string=lambda t: isinstance(t, Comment)):
        text = str(c).strip()
        if "NewPP limit report" in text or "Transclusion expansion" in text or "Saved in parser cache" in text:
            c.extract()
        elif text.startswith("Added by HTTrack") or text.startswith("/Added by HTTrack"):
            c.extract()

    toc = root2.select_one("#toc")
    if toc:
        toc.decompose()

    for pf in root2.select(".printfooter"):
        pf.decompose()

    text = root2.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return canonical, page_title, text


def _have_lxml() -> bool:
    try:
        import lxml  
        return True
    except ImportError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Важный текст из одной HTML-страницы MoodleDocs.")
    p.add_argument("html_file", type=Path, help="Путь к .html")
    args = p.parse_args()
    path: Path = args.html_file
    if not path.is_file():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    raw = path.read_text(encoding="utf-8", errors="replace")
    canonical, title, body = extract_important_text(raw)

    lines = []
    if canonical:
        lines.append(f"URL: {canonical}")
    if title:
        lines.append(f"Title: {title}")
    lines.append("")
    lines.append(body if body else "(не найден блок содержимого)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
