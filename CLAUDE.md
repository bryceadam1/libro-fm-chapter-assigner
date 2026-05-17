# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`libro-assigner` is a Python CLI tool that embeds proper chapter markers into Libro.fm audiobook M4B files. It takes the M4B file and a user-supplied list of section headings, then uses OpenAI's Whisper model to assign each heading to the audio track it most likely came from.

## Setup & Commands

**Package manager:** `uv` (Python 3.14)

```bash
uv sync                  # Install dependencies into .venv
uv run libro-assign --help
```

**System dependency required:** `ffmpeg` and `ffprobe` must be on PATH.

**Run the tool:**
```bash
# GUI (recommended)
uv run libro-gui

# CLI: extract headings from an EPUB TOC
uv run libro-assign extract-toc book.epub

# CLI: assign headings to an M4B
uv run libro-assign assign book.m4b "Introduction" "Chapter One" "Chapter Two" "Epilogue"
uv run libro-assign assign book.m4b --model small --dry-run "Prologue" "Part I" "Part II"
```

There are no tests or linting configs in this project yet.

## Architecture

### Core idea — Whisper perplexity scoring

Rather than transcribing audio and fuzzy-matching against a TOC, the new approach uses Whisper's decoder log-probabilities directly:

1. The M4B file's existing track boundaries are read to get one audio segment per track.
2. Each audio segment is encoded by Whisper's encoder.
3. For every candidate section heading, Whisper's decoder is run in **forced-decoding** mode: the heading tokens are fed one-by-one as decoder targets conditioned on the audio, and the per-token log probabilities are collected.
4. The **mean log probability** across all heading tokens is the heading's score for that track (higher = more likely the audio is saying that heading).
5. The heading with the highest mean log probability is assigned to each track.
6. Assigned chapter markers are written back into the M4B with ffmpeg (no re-encoding).

### Modules

[toc.py](src/libro_assigner/toc.py): Parses an EPUB file with `ebooklib` and returns a flat, ordered list of section heading strings from its table of contents. Handles both EPUB 2 (NCX) and EPUB 3 (nav document) formats. Nested TOC entries are flattened depth-first, preserving document order.

[extractor.py](src/libro_assigner/extractor.py): Calls `ffprobe` to read existing chapter/track boundaries and total duration from the input M4B. Returns a `Chapter` list with `start_ms`/`end_ms`.

[scorer.py](src/libro_assigner/scorer.py): Core scoring engine. Loads a Whisper model and tokenizer (`load_model`), then for each track extracts a temporary audio segment and calls `score_headings`, which encodes the audio once and runs one decoder forward pass per candidate heading. The public entry point is `assign_chapters`, which drives the full loop and returns `AlignedChapter` results.

[writer.py](src/libro_assigner/writer.py): Builds an ffmetadata file from the final `AlignedChapter` list and calls `ffmpeg -codec copy` to embed chapters without re-encoding.

[cli.py](src/libro_assigner/cli.py): Click entry point. `assign` reads existing track boundaries, calls `scorer.assign_chapters`, prints results, and optionally writes the output M4B.

[gui.py](src/libro_assigner/gui.py): PySide6 GUI (`libro-gui` entry point). `MainWindow` has three states: pre-import (empty tree), post-import (tracks in left column, headings in right column, both listed independently), and post-alignment (heading-indexed rows where each EPUB heading occupies one row and its assigned track fills the left cell, or is blank if no track was assigned; duplicate-assigned tracks overflow to the bottom with a blank heading cell). Assignment runs in a `_Worker` `QObject` moved to a `QThread`; progress is reported back to the main thread via Qt signals.

### Key data types

- `Chapter` ([extractor.py](src/libro_assigner/extractor.py)): Existing track from the source file (`index`, `title`, `start_ms`, `end_ms`)
- `AlignedChapter` ([types.py](src/libro_assigner/types.py)): Final result passed to the writer (`title`, `start_ms`)

### Whisper forced-decoding details

- Default model: `tiny`. Any Whisper size (tiny, base, small, medium, large) selectable with `--model`. Models download automatically to `~/.cache/whisper/` on first use.
- Each track's audio is extracted to a temp file with `ffmpeg -codec copy` (no re-encode), trimmed/padded to 30 s by `whisper.pad_or_trim`, then passed through `model.encoder` once per track.
- The decoder prefix is `tokenizer.sot_sequence_including_notimestamps` — `[<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]`. Heading tokens follow immediately.
- Decoder output at position `j` gives `P(token_{j+1} | preceding tokens, audio)`, so heading token `i` is read from logit position `prefix_len - 1 + i`.
- Each track picks its best heading independently; the same heading can be assigned to multiple tracks.
