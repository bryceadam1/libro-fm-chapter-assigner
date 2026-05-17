"""Extract section headings from an EPUB table of contents."""

from __future__ import annotations

from pathlib import Path

from ebooklib import epub


def extract_toc(epub_path: Path) -> list[str]:
    """Return an ordered flat list of section heading strings from the EPUB TOC."""
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": False})
    return _walk_toc(book.toc)


def _walk_toc(items) -> list[str]:
    result = []
    for item in items:
        if isinstance(item, epub.Link):
            if item.title:
                result.append(item.title.strip())
        elif isinstance(item, tuple):
            section, children = item
            if hasattr(section, "title") and section.title:
                result.append(section.title.strip())
            result.extend(_walk_toc(children))
    return result
