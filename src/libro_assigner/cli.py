"""Command-line interface for libro-fm-chapter-assigner."""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
def main() -> None:
    """Assign chapter names to Libro.fm audiobook files."""


@main.command("extract-toc")
@click.argument("epub_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def extract_toc(epub_file: Path) -> None:
    """Print section headings from EPUB_FILE's table of contents, one per line."""
    from .toc import extract_toc as _extract_toc

    headings = _extract_toc(epub_file)
    for heading in headings:
        click.echo(heading)


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
    "--seconds",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds of audio sampled from the start of each track for scoring.",
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
    seconds: float,
    dry_run: bool,
) -> None:
    """Assign HEADINGS to tracks in INPUT_FILE using Whisper perplexity scoring.

    HEADINGS is an ordered list of section heading strings. Each audio track in
    INPUT_FILE is scored against every heading; the heading with the highest mean
    log probability (given that track's audio) is assigned to that track.
    """
    from .extractor import extract_chapters, get_duration_ms
    from .scorer import assign_chapters
    from .writer import write_chapters

    if output is None:
        output = input_file.with_stem(input_file.stem + "_chaptered").with_suffix(".m4b")

    click.echo(f"Reading existing chapters from {input_file.name} …")
    chapters = extract_chapters(input_file)
    if not chapters:
        click.echo(
            "  No chapter markers found. The input file must have existing track boundaries.",
            err=True,
        )
        sys.exit(1)

    total_ms = get_duration_ms(input_file)
    click.echo(f"  Found {len(chapters)} track(s), duration {total_ms / 1000:.0f}s")
    click.echo(f"  Headings ({len(headings)}): {list(headings)}")

    click.echo(f"\nScoring {len(chapters)} track(s) × {len(headings)} heading(s) …")
    aligned = assign_chapters(
        chapters=chapters,
        m4b_path=input_file,
        headings=list(headings),
        model_name=model,
        seconds=seconds,
    )

    click.echo(f"\nAssignments ({len(aligned)} tracks):")
    for ch in aligned:
        mins, secs = divmod(ch.start_ms // 1000, 60)
        hrs, mins = divmod(mins, 60)
        click.echo(f"  [{hrs:02d}:{mins:02d}:{secs:02d}]  {ch.title}")

    if dry_run:
        click.echo("\n(Dry run — no file written.)")
        return

    click.echo(f"\nWriting {output.name} …")
    write_chapters(input_file, output, aligned, total_ms)
    click.echo(f"Done. Output: {output}")
