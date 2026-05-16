"""Command-line interface for libro-fm-chapter-assigner."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def main() -> None:
    """Assign chapter names to Libro.fm audiobook files."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("headings", nargs=-1, required=True)
@click.option(
    "--model",
    default="tiny",
    show_default=True,
    help="Whisper model size (tiny, base, small, medium, large).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path. Defaults to <input>_chaptered.m4b.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print assignments without writing an output file.",
)
def assign(
    input_file: Path,
    headings: tuple[str, ...],
    model: str,
    output: Path | None,
    dry_run: bool,
) -> None:
    """Assign HEADINGS to tracks in INPUT_FILE using Whisper perplexity scoring.

    HEADINGS is an ordered list of section heading strings. Each audio track in
    INPUT_FILE is scored against every heading; the heading with the highest mean
    log probability (given that track's audio) is assigned to that track.
    """
    raise NotImplementedError("Not yet implemented.")
