"""Read existing chapter markers and audio duration from an M4B file via ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chapter:
    index: int
    title: str
    start_ms: int
    end_ms: int


def extract_chapters(path: Path) -> list[Chapter]:
    """Return the list of chapters currently embedded in the file.

    Returns an empty list if the file has no chapter markers.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_chapters",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    chapters = []
    for i, ch in enumerate(data.get("chapters", [])):
        # ffprobe reports start_time / end_time in seconds (as strings)
        start_ms = int(float(ch["start_time"]) * 1000)
        end_ms = int(float(ch["end_time"]) * 1000)
        title = ch.get("tags", {}).get("title", f"Track {i + 1:02d}")
        chapters.append(Chapter(index=i, title=title, start_ms=start_ms, end_ms=end_ms))
    return chapters


def get_duration_ms(path: Path) -> int:
    """Return the total audio duration of the file in milliseconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    duration_sec = float(data["format"]["duration"])
    return int(duration_sec * 1000)
