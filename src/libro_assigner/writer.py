"""Write chapter metadata into an M4B file using ffmpeg (no re-encoding)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .aligner import AlignedChapter


def _escape_ffmeta(value: str) -> str:
    """Escape special characters for the ffmetadata format."""
    # = ; # \ must be escaped with a backslash
    return re.sub(r"([=;#\\])", r"\\\1", value)


import re  # noqa: E402  (placed after function to avoid forward-ref issues)


def build_ffmetadata(
    chapters: list[AlignedChapter],
    total_duration_ms: int,
    title: str = "",
    author: str = "",
) -> str:
    """Build the content of an ffmetadata file from the chapter list."""
    lines = [";FFMETADATA1"]
    if title:
        lines.append(f"title={_escape_ffmeta(title)}")
    if author:
        lines.append(f"artist={_escape_ffmeta(author)}")
    lines.append("")

    for i, ch in enumerate(chapters):
        start_ms = ch.start_ms
        # Chapter end = next chapter's start, or total duration for the last one
        if i + 1 < len(chapters):
            end_ms = chapters[i + 1].start_ms
        else:
            end_ms = total_duration_ms

        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_escape_ffmeta(ch.title)}",
            "",
        ]

    return "\n".join(lines)


def write_chapters(
    input_path: Path,
    output_path: Path,
    chapters: list[AlignedChapter],
    total_duration_ms: int,
    title: str = "",
    author: str = "",
) -> None:
    """Embed *chapters* into *input_path* and write result to *output_path*.

    Uses ``ffmpeg -codec copy`` — no audio re-encoding.
    Raises ``subprocess.CalledProcessError`` on ffmpeg failure.
    """
    metadata = build_ffmetadata(chapters, total_duration_ms, title=title, author=author)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ffmeta", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(metadata)
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",                   # overwrite output without asking
                "-i", str(input_path),
                "-i", str(tmp_path),
                "-map_metadata", "1",   # use the metadata file
                "-map_chapters", "1",   # use chapters from metadata file
                "-codec", "copy",       # no re-encode
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
