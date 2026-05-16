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

[extractor.py](src/libro_assigner/extractor.py): Calls `ffprobe` to read existing chapter/track boundaries and total duration from the input M4B. Returns a `Chapter` list with `start_ms`/`end_ms`.

[writer.py](src/libro_assigner/writer.py): Builds an ffmetadata file from the final `AlignedChapter` list and calls `ffmpeg -codec copy` to embed chapters without re-encoding.

[cli.py](src/libro_assigner/cli.py): Click entry point. `assign` command accepts the M4B path and a variadic `HEADINGS` argument. Not yet implemented beyond the stub.

### Key data types

- `Chapter` ([extractor.py](src/libro_assigner/extractor.py)): Existing track from the source file (`index`, `title`, `start_ms`, `end_ms`)
- `AlignedChapter` — to be defined in a scorer/aligner module: final result (`title`, `start_ms`)

### Whisper forced-decoding notes

- Default model: `tiny`. Any Whisper size (tiny, base, small, medium, large) can be selected with `--model`.
- Whisper models are downloaded automatically by `openai-whisper` on first use and cached in `~/.cache/whisper/`.
- Audio is extracted per-track from the M4B using ffmpeg before being passed to Whisper's encoder.
- The Whisper tokenizer prepends the standard prompt tokens (language, task) before the heading tokens; these should be excluded when computing the mean log probability.
