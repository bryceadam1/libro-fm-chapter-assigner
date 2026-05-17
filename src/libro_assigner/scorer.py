"""Assign section headings to audio tracks via Whisper forced-decoding perplexity.

For each audio track, every candidate heading is scored by running Whisper's
decoder in forced-decoding mode: the heading tokens are fed as decoder targets
conditioned on the track's audio, and the mean log probability across all heading
tokens is recorded.  The heading with the highest mean log probability wins.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
import whisper
import whisper.tokenizer

from .extractor import Chapter
from .types import AlignedChapter


def load_model(model_name: str = "tiny") -> tuple:
    """Load a Whisper model and a matching English-transcription tokenizer."""
    model = whisper.load_model(model_name)
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )
    return model, tokenizer


def _extract_audio_segment(
    source: Path, start_ms: int, end_ms: int, output: Path, max_seconds: float = 30.0
) -> None:
    """Extract a time slice of the M4B to a temp file via ffmpeg."""
    duration = min((end_ms - start_ms) / 1000, max_seconds)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_ms / 1000),
            "-t", str(duration),
            "-i", str(source),
            "-vn",
            "-codec:a", "copy",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def _encode_audio(model, audio_path: Path, seconds: float = 30.0) -> torch.Tensor:
    """Load audio, compute mel spectrogram, and return encoder output."""
    audio = whisper.load_audio(str(audio_path))
    # Truncate to the requested window, then pad to the full 30-s shape the
    # encoder requires (N_SAMPLES = 480000). Audio after `seconds` becomes zeros.
    audio = audio[:int(seconds * whisper.audio.SAMPLE_RATE)]
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    with torch.no_grad():
        return model.encoder(mel.unsqueeze(0))


def mean_log_prob(
    model,
    tokenizer,
    audio_features: torch.Tensor,
    heading: str,
) -> float:
    """Return mean log P(heading tokens | audio) via one decoder forward pass."""
    tokens = tokenizer.encode(heading)
    if not tokens:
        return float("-inf")

    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    input_ids = torch.tensor(
        [prefix + tokens], device=model.device, dtype=torch.long
    )

    with torch.no_grad():
        logits = model.decoder(input_ids, audio_features)

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    prefix_len = len(prefix)
    token_log_probs = [
        log_probs[prefix_len - 1 + i, tok].item()
        for i, tok in enumerate(tokens)
    ]
    return sum(token_log_probs) / len(token_log_probs)


def _batch_score_headings(
    model,
    tokenizer,
    audio_features: torch.Tensor,
    headings: list[str],
) -> list[tuple[str, float]]:
    """Score all headings in a single batched decoder forward pass.

    All headings are padded to the same token length, stacked into one batch,
    and the decoder is called once.  This replaces the previous per-heading
    loop and gives a large speedup (especially on GPU).

    Audio features are broadcast across the batch via expand so no extra
    memory copy is needed.
    """
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    prefix_len = len(prefix)

    all_tokens = [(h, tokenizer.encode(h)) for h in headings]
    valid = [(h, toks) for h, toks in all_tokens if toks]

    if not valid:
        return [(h, float("-inf")) for h in headings]

    max_len = max(len(toks) for _, toks in valid)
    eot = tokenizer.eot

    rows = [prefix + toks + [eot] * (max_len - len(toks)) for _, toks in valid]
    input_ids = torch.tensor(rows, device=model.device, dtype=torch.long)

    # Broadcast the single audio encoding across the heading batch
    xa = audio_features.expand(len(rows), -1, -1)

    with torch.no_grad():
        logits = model.decoder(input_ids, xa)

    # logits: (n_valid, prefix_len + max_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    valid_scores: dict[str, float] = {}
    for i, (h, toks) in enumerate(valid):
        token_log_probs = [
            log_probs[i, prefix_len - 1 + j, tok].item()
            for j, tok in enumerate(toks)
        ]
        valid_scores[h] = sum(token_log_probs) / len(token_log_probs)

    scores = [(h, valid_scores.get(h, float("-inf"))) for h in headings]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def score_headings(
    model,
    tokenizer,
    audio_path: Path,
    headings: list[str],
    seconds: float = 30.0,
) -> list[tuple[str, float]]:
    """Return (heading, mean_log_prob) pairs sorted best-first for *audio_path*."""
    audio_features = _encode_audio(model, audio_path, seconds=seconds)
    return _batch_score_headings(model, tokenizer, audio_features, headings)


def _percentile(values: list[float], p: float) -> float:
    """p-th percentile of *values* (p in [0, 1]) via linear interpolation."""
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def _run_assignment(
    all_ranked: list[list[tuple[str, float]]],
) -> dict[int, tuple[str, float]]:
    """Apply 90th-percentile threshold then greedy best-first matching.

    Returns a dict mapping track index → (heading, score) for every track
    that received an assignment.  Operates on however many tracks have been
    scored so far, so it can be called incrementally after each new track.
    """
    eligible: list[tuple[int, str, float]] = []
    for i, ranked in enumerate(all_ranked):
        if not ranked:
            continue
        threshold = _percentile([s for _, s in ranked], 0.90)
        for h, s in ranked:
            if s >= threshold:
                eligible.append((i, h, s))

    eligible.sort(key=lambda x: x[2], reverse=True)
    assigned: dict[int, tuple[str, float]] = {}
    claimed: set[str] = set()
    for track_idx, heading, score in eligible:
        if track_idx not in assigned and heading not in claimed:
            assigned[track_idx] = (heading, score)
            claimed.add(heading)

    return assigned


def assign_chapters(
    chapters: list[Chapter],
    m4b_path: Path,
    headings: list[str],
    model_name: str = "tiny",
    seconds: float = 5.0,
    progress_callback: Callable[[int, int], None] | None = None,
    on_track_done: Callable[[int, str], None] | None = None,
    on_assignments_updated: Callable[[dict[int, str]], None] | None = None,
) -> list[AlignedChapter]:
    """Score every track against every heading and return the winning assignments.

    Scoring — for each track, all headings are scored in a single batched
    decoder forward pass (one encoder pass + one decoder pass per track).

    After each track is scored, _run_assignment is called on all tracks scored
    so far: it applies a per-track 90th-percentile threshold, then does a
    global greedy best-first one-to-one match.  on_assignments_updated is
    called with the current dict[track_index, heading] so the GUI can show
    assignments appearing (and potentially shifting) in real time.

    on_track_done(track_index, heading) is kept for CLI/legacy callers; it
    receives the track's current best heading after each intermediate match.
    """
    model, tokenizer = load_model(model_name)
    total = len(chapters)
    all_ranked: list[list[tuple[str, float]]] = []
    current_assigned: dict[int, tuple[str, float]] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, chapter in enumerate(chapters):
            chunk_path = tmp / f"track_{chapter.index:04d}.m4a"
            _extract_audio_segment(
                m4b_path, chapter.start_ms, chapter.end_ms, chunk_path,
                max_seconds=seconds,
            )
            ranked = score_headings(model, tokenizer, chunk_path, headings, seconds=seconds)
            all_ranked.append(ranked)

            current_assigned = _run_assignment(all_ranked)

            if on_assignments_updated is not None:
                on_assignments_updated(
                    {j: h for j, (h, _) in current_assigned.items()}
                )
            if on_track_done is not None:
                h, _ = current_assigned.get(i, ("", 0.0))
                on_track_done(i, h)
            if progress_callback is not None:
                progress_callback(i + 1, total)

    # Print final summary
    for i, chapter in enumerate(chapters):
        best_heading, best_score = current_assigned.get(i, ("", float("-inf")))
        if best_heading:
            print(
                f"[scorer] Track {chapter.index:02d}: {best_heading!r} "
                f"(mean log prob {best_score:.3f})",
                file=sys.stderr,
            )
        else:
            print(f"[scorer] Track {chapter.index:02d}: unassigned", file=sys.stderr)

    return [
        AlignedChapter(
            title=current_assigned.get(i, ("", 0.0))[0],
            start_ms=chapter.start_ms,
        )
        for i, chapter in enumerate(chapters)
    ]
