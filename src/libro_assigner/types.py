"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignedChapter:
    title: str
    start_ms: int
