"""PySide6 GUI for libro-fm-chapter-assigner."""

from __future__ import annotations

import bisect
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .extractor import extract_chapters
from .toc import extract_toc

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Column indices
_COL_PLAY     = 0   # narrow ▶/■ button column
_COL_TRACK    = 1
_COL_ASSIGNED = 2   # heading lands here when mapped
_COL_ARROW    = 3   # narrow ←/→ arrow button between the two heading columns
_COL_PENDING  = 4   # headings start here; stays here if unassigned

# Item data roles (stored in column 0 of each item)
_ROLE_TRACK_IDX   = Qt.ItemDataRole.UserRole       # int | None
_ROLE_HEADING_POS = Qt.ItemDataRole.UserRole + 1   # int | None

_PREVIEW_SECONDS = 30
_PLAYBACK_POLL_MS = 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_track(ch) -> str:
    start = ch.start_ms // 1000
    h, rem = divmod(start, 3600)
    m, s = divmod(rem, 60)
    dur_min = (ch.end_ms - ch.start_ms) / 60_000
    return f"Track {ch.index + 1:03d}   [{h:02d}:{m:02d}:{s:02d}]   ({dur_min:.0f} min)"


def _separator(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ---------------------------------------------------------------------------
# Out-of-order detection
# ---------------------------------------------------------------------------

def _out_of_order_tracks(
    track_assignments: dict[int, str],
    headings: list[str],
) -> set[int]:
    """Return track indices whose heading assignment breaks the expected order.

    Sorts assigned tracks by track index, maps each to its heading's first
    position in the headings list, then finds the Longest Increasing
    Subsequence of those positions.  Any track NOT in the LIS is out of order.
    """
    heading_first_pos: dict[str, int] = {}
    for i, h in enumerate(headings):
        if h not in heading_first_pos:
            heading_first_pos[h] = i

    pairs = sorted(
        (ti, heading_first_pos[h])
        for ti, h in track_assignments.items()
        if h and h in heading_first_pos
    )
    if not pairs:
        return set()

    h_values = [p for _, p in pairs]

    tail_vals: list[int] = []
    tail_src:  list[int] = []   # index into pairs for each pile's current tail
    predecessor = [-1] * len(pairs)

    for i, v in enumerate(h_values):
        lo = bisect.bisect_left(tail_vals, v)
        if lo == len(tail_vals):
            tail_vals.append(v)
            tail_src.append(i)
        else:
            tail_vals[lo] = v
            tail_src[lo] = i
        predecessor[i] = tail_src[lo - 1] if lo > 0 else -1

    lis: set[int] = set()
    idx = tail_src[-1]
    while idx != -1:
        lis.add(idx)
        idx = predecessor[idx]

    return {pairs[i][0] for i in range(len(pairs)) if i not in lis}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _zip_unassigned_blocks(
    rows: list[tuple[int | None, int | None]],
) -> list[tuple[int | None, int | None]]:
    """Merge adjacent runs of unassigned-track and unassigned-heading rows.

    When a run of (track_idx, None) is immediately next to a run of
    (None, h_pos) — in either order — the two runs are zipped together so
    that one row contains both a track and a heading.  The surplus from the
    longer run becomes standalone rows at the end of the merged block.
    The track and heading remain unassigned; this is purely a display layout.
    """
    result: list[tuple[int | None, int | None]] = []
    i = 0
    while i < len(rows):
        t, h = rows[i]

        if t is not None and h is None:
            j = i
            while j < len(rows) and rows[j][0] is not None and rows[j][1] is None:
                j += 1
            track_run = [rows[k][0] for k in range(i, j)]

            k = j
            while k < len(rows) and rows[k][0] is None and rows[k][1] is not None:
                k += 1
            head_run = [rows[m][1] for m in range(j, k)]

            n = min(len(track_run), len(head_run))
            for p in range(n):
                result.append((track_run[p], head_run[p]))
            for p in range(n, len(track_run)):
                result.append((track_run[p], None))
            for p in range(n, len(head_run)):
                result.append((None, head_run[p]))
            i = k

        elif t is None and h is not None:
            j = i
            while j < len(rows) and rows[j][0] is None and rows[j][1] is not None:
                j += 1
            head_run = [rows[k][1] for k in range(i, j)]

            k = j
            while k < len(rows) and rows[k][0] is not None and rows[k][1] is None:
                k += 1
            track_run = [rows[m][0] for m in range(j, k)]

            n = min(len(track_run), len(head_run))
            for p in range(n):
                result.append((track_run[p], head_run[p]))
            for p in range(n, len(head_run)):
                result.append((None, head_run[p]))
            for p in range(n, len(track_run)):
                result.append((track_run[p], None))
            i = k

        else:
            result.append((t, h))
            i += 1

    return result


# ---------------------------------------------------------------------------
# Drag-and-drop chapter tree
# ---------------------------------------------------------------------------

class _ChapterTree(QTreeWidget):
    """QTreeWidget with drag-and-drop heading reordering and assignment.

    Every drop emits a single ``heading_dropped`` signal carrying the full
    context so the handler can perform reorder + assign/unassign atomically:

        heading_dropped(src_h_pos, src_t_idx, dst_h_pos, dst_t_idx, col, drop_above)

    src_t_idx / dst_h_pos / dst_t_idx are ``int | None`` (carried as object).
    """

    heading_dropped = Signal(int, object, object, object, int, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)

    def _dragged_h_pos(self) -> int | None:
        item = self.currentItem()
        return None if item is None else item.data(0, _ROLE_HEADING_POS)

    def _dragged_t_idx(self) -> int | None:
        item = self.currentItem()
        return None if item is None else item.data(0, _ROLE_TRACK_IDX)

    def _classify(self, event) -> tuple[bool, bool, bool, object, object, bool]:
        """Return (reorder, assign, unassign, dst_h_pos, dst_t_idx, drop_above)."""
        pos      = event.position().toPoint()
        col      = self.columnAt(pos.x())
        src_h    = self._dragged_h_pos()
        src_t    = self._dragged_t_idx()
        dst      = self.itemAt(pos)
        dst_h    = dst.data(0, _ROLE_HEADING_POS) if dst is not None else None
        dst_t    = dst.data(0, _ROLE_TRACK_IDX)   if dst is not None else None
        if dst_h == src_h:
            dst_h = None  # same heading → no reorder
        drop_above = (
            self.dropIndicatorPosition()
            == QAbstractItemView.DropIndicatorPosition.AboveItem
        )
        will_reorder  = dst_h is not None
        will_assign   = col == _COL_ASSIGNED and dst_t is not None
        will_unassign = col == _COL_PENDING  and src_t is not None
        return will_reorder, will_assign, will_unassign, dst_h, dst_t, drop_above

    def dragEnterEvent(self, event) -> None:
        if self._dragged_h_pos() is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._dragged_h_pos() is None:
            event.ignore()
            return
        reorder, assign, unassign, dst_h, _dst_t, _above = self._classify(event)
        if not (reorder or assign or unassign):
            event.ignore()
            return
        # Show the position indicator only for a pure reorder.
        if reorder and not assign and not unassign:
            super().dragMoveEvent(event)
        else:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        src_h_pos = self._dragged_h_pos()
        if src_h_pos is None:
            event.ignore()
            return
        reorder, assign, unassign, dst_h, dst_t, drop_above = self._classify(event)
        if not (reorder or assign or unassign):
            event.ignore()
            return
        event.accept()
        self.heading_dropped.emit(
            src_h_pos, self._dragged_t_idx(), dst_h, dst_t,
            self.columnAt(event.position().toPoint().x()), drop_above,
        )

    def mouseDoubleClickEvent(self, event) -> None:
        pos = event.position().toPoint()
        item = self.itemAt(pos)
        if item is not None and item.data(0, _ROLE_HEADING_POS) is not None:
            # Edit the column that currently holds the heading text.
            edit_col = _COL_ASSIGNED if item.text(_COL_ASSIGNED) else _COL_PENDING
            self.editItem(item, edit_col)
            return
        super().mouseDoubleClickEvent(event)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _Worker(QObject):
    progress            = Signal(int, int)  # (completed, total)
    assignments_updated = Signal(object)    # dict[int, str] after each track
    finished            = Signal()
    error               = Signal(str)

    def __init__(
        self,
        chapters: list,
        m4b_path: Path,
        headings: list[str],
        model_name: str,
        seconds: int,
    ) -> None:
        super().__init__()
        self._chapters   = chapters
        self._m4b_path   = m4b_path
        self._headings   = headings
        self._model_name = model_name
        self._seconds    = seconds

    def run(self) -> None:
        from .scorer import assign_chapters

        try:
            assign_chapters(
                chapters=self._chapters,
                m4b_path=self._m4b_path,
                headings=self._headings,
                model_name=self._model_name,
                seconds=float(self._seconds),
                on_assignments_updated=lambda a: self.assignments_updated.emit(dict(a)),
                progress_callback=lambda done, total: self.progress.emit(done, total),
            )
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Libro.fm Chapter Assigner")
        self.resize(1280, 720)

        self._m4b_path: Path | None = None
        self._epub_path: Path | None = None
        self._chapters: list = []
        self._headings: list[str] = []

        # Assignment state: maps chapter-list-index → winning heading string.
        self._track_assignments: dict[int, str] = {}
        self._claimed_headings: set[str] = set()

        # Playback state
        self._playback_proc: subprocess.Popen | None = None
        self._active_play_btn: QPushButton | None = None
        self._active_chapter_start_ms: int | None = None
        self._play_buttons: dict[int, QPushButton] = {}  # start_ms → button

        # Guard flag: True while _rebuild_view is running so itemChanged
        # signals from programmatic setText calls are ignored.
        self._rebuilding = False

        # Worker / thread
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

        self._build_ui()

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(_PLAYBACK_POLL_MS)
        self._playback_timer.timeout.connect(self._check_playback)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        root.addLayout(self._build_import_row())
        root.addWidget(_separator(central))
        root.addLayout(self._build_control_row())
        root.addWidget(_separator(central))
        root.addWidget(self._build_tree(), stretch=1)

    def _build_import_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._import_m4b_btn = QPushButton("Import M4B…")
        self._import_m4b_btn.clicked.connect(self._import_m4b)
        row.addWidget(self._import_m4b_btn)

        self._m4b_label = QLabel("No file selected")
        self._m4b_label.setStyleSheet("color: gray")
        row.addWidget(self._m4b_label)

        row.addStretch()

        self._epub_label = QLabel("No file selected")
        self._epub_label.setStyleSheet("color: gray")
        row.addWidget(self._epub_label)

        self._import_epub_btn = QPushButton("Import EPUB…")
        self._import_epub_btn.clicked.connect(self._import_epub)
        row.addWidget(self._import_epub_btn)

        return row

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._assign_btn = QPushButton("Assign Chapters")
        self._assign_btn.setEnabled(False)
        self._assign_btn.clicked.connect(self._start_assign)
        row.addWidget(self._assign_btn)

        row.addWidget(QLabel("Model:"))

        self._model_combo = QComboBox()
        self._model_combo.addItems(_WHISPER_MODELS)
        self._model_combo.setFixedWidth(90)
        row.addWidget(self._model_combo)

        row.addWidget(QLabel("Seconds:"))

        self._seconds_spin = QSpinBox()
        self._seconds_spin.setRange(1, 30)
        self._seconds_spin.setValue(5)
        self._seconds_spin.setFixedWidth(55)
        self._seconds_spin.setToolTip("Seconds of audio sampled from each track for scoring")
        row.addWidget(self._seconds_spin)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        row.addWidget(self._progress, stretch=1)

        self._status_label = QLabel("")
        row.addWidget(self._status_label)

        row.addStretch()

        self._export_btn = QPushButton("Export .m4b…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        row.addWidget(self._export_btn)

        return row

    def _build_tree(self) -> _ChapterTree:
        tree = _ChapterTree()
        tree.setColumnCount(5)
        tree.setHeaderLabels([
            "",
            "Track",
            "Chapter Heading",
            "",
            "Unassigned Chapter Headings",
        ])
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        hdr = tree.header()
        hdr.setSectionResizeMode(_COL_PLAY,     QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(_COL_TRACK,    QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_COL_ASSIGNED, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_COL_ARROW,    QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(_COL_PENDING,  QHeaderView.ResizeMode.Interactive)
        tree.setColumnWidth(_COL_PLAY,     36)
        tree.setColumnWidth(_COL_TRACK,    300)
        tree.setColumnWidth(_COL_ASSIGNED, 300)
        tree.setColumnWidth(_COL_ARROW,    36)
        tree.setColumnWidth(_COL_PENDING,  300)

        bold = QFont()
        bold.setBold(True)
        hdr.setFont(bold)

        tree.heading_dropped.connect(self._on_drop)
        tree.itemChanged.connect(self._on_item_changed)
        tree.customContextMenuRequested.connect(self._on_context_menu)

        self._tree = tree
        return tree

    # ── Playback ────────────────────────────────────────────────────────────

    def _make_play_btn(self, chapter) -> QPushButton:
        btn = QPushButton("▶")
        btn.setFixedSize(28, 22)
        btn.setToolTip(f"Preview first {_PREVIEW_SECONDS}s of this track")
        btn.clicked.connect(lambda: self._toggle_play(btn, chapter))
        return btn

    def _toggle_play(self, btn: QPushButton, chapter) -> None:
        is_running = (
            self._playback_proc is not None
            and self._playback_proc.poll() is None
        )

        if is_running:
            was_this_btn = (self._active_play_btn is btn)
            if self._active_play_btn is not None:
                self._active_play_btn.setText("▶")
            self._playback_proc.terminate()
            self._playback_proc = None
            self._active_play_btn = None
            self._active_chapter_start_ms = None
            self._playback_timer.stop()
            if was_this_btn:
                return

        self._playback_proc = subprocess.Popen(
            [
                "ffplay", "-nodisp", "-autoexit",
                "-ss", str(chapter.start_ms / 1000),
                "-t", str(_PREVIEW_SECONDS),
                str(self._m4b_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        btn.setText("■")
        self._active_play_btn = btn
        self._active_chapter_start_ms = chapter.start_ms
        self._playback_timer.start()

    def _check_playback(self) -> None:
        if self._playback_proc is not None and self._playback_proc.poll() is not None:
            self._playback_proc = None
            if self._active_play_btn is not None:
                try:
                    self._active_play_btn.setText("▶")
                except RuntimeError:
                    pass
            self._active_play_btn = None
            self._active_chapter_start_ms = None
            self._playback_timer.stop()

    def _stop_playback(self) -> None:
        self._playback_timer.stop()
        if self._playback_proc is not None and self._playback_proc.poll() is None:
            self._playback_proc.terminate()
        self._playback_proc = None
        if self._active_play_btn is not None:
            try:
                self._active_play_btn.setText("▶")
            except RuntimeError:
                pass
        self._active_play_btn = None
        self._active_chapter_start_ms = None

    # ── Tree rendering ───────────────────────────────────────────────────────

    def _rebuild_view(self) -> None:
        self._rebuilding = True
        active_start_ms = self._active_chapter_start_ms
        scroll_pos = self._tree.verticalScrollBar().value()
        self._playback_timer.stop()
        self._tree.clear()
        self._play_buttons = {}
        self._active_play_btn = None

        # Map heading string → ordered list of positions in self._headings.
        # Duplicate heading strings (e.g. "Chapter 48" in Deleted Scenes) get
        # separate entries so each occurrence can be tracked independently.
        heading_positions: dict[str, list[int]] = defaultdict(list)
        for i, h in enumerate(self._headings):
            heading_positions[h].append(i)

        assigned_tracks = set(self._track_assignments)
        claimed_strings = set(self._track_assignments.values())

        # Rows are (track_idx | None, heading_pos | None) where heading_pos is
        # the index into self._headings (not the string itself), so duplicate
        # headings sort by their respective positions independently.
        rows: list[tuple[int | None, int | None]] = []

        for track_idx, heading in self._track_assignments.items():
            rows.append((track_idx, heading_positions[heading][0]))

        for i in range(len(self._chapters)):
            if i not in assigned_tracks:
                rows.append((i, None))

        for h, positions in heading_positions.items():
            # Skip the first occurrence if this heading string is claimed;
            # any additional occurrences (duplicates) remain as unassigned rows.
            start = 1 if h in claimed_strings else 0
            for pos in positions[start:]:
                rows.append((None, pos))

        # Iteratively stabilize: sort by heading pos (nulls inherit preceding),
        # then by track (nulls inherit preceding), until stable or 10 passes.
        for _ in range(10):
            prev = rows[:]

            last: float = float('-inf')
            h_keys: list[float] = []
            for _, h_pos in rows:
                if h_pos is not None:
                    last = float(h_pos)
                h_keys.append(last)
            rows = [r for _, r in sorted(enumerate(rows), key=lambda x: h_keys[x[0]])]

            last = float('-inf')
            t_keys: list[float] = []
            for t_idx, _ in rows:
                if t_idx is not None:
                    last = float(t_idx)
                t_keys.append(last)
            rows = [r for _, r in sorted(enumerate(rows), key=lambda x: t_keys[x[0]])]

            if rows == prev:
                break

        rows = _zip_unassigned_blocks(rows)

        out_of_order = _out_of_order_tracks(self._track_assignments, self._headings)
        yellow = QColor(255, 255, 160)
        black  = QColor(0, 0, 0)

        # Build per-track out-of-order tooltips.
        oo_tooltips: dict[int, str] = {}
        if out_of_order:
            pairs = sorted(
                (ti, h) for ti, h in self._track_assignments.items() if h
            )
            for i, (ti, h) in enumerate(pairs):
                if ti not in out_of_order:
                    continue
                prev_ok = next(
                    ((tj, hj) for tj, hj in reversed(pairs[:i]) if tj not in out_of_order),
                    None,
                )
                next_ok = next(
                    ((tj, hj) for tj, hj in pairs[i + 1:] if tj not in out_of_order),
                    None,
                )
                msg = f'"{h}" is out of order'
                if prev_ok and next_ok:
                    msg += f' — expected between "{prev_ok[1]}" and "{next_ok[1]}"'
                elif prev_ok:
                    msg += f' — expected after "{prev_ok[1]}"'
                elif next_ok:
                    msg += f' — expected before "{next_ok[1]}"'
                oo_tooltips[ti] = msg

        # Render rows into the tree.
        for t_idx, h_pos in rows:
            chapter = self._chapters[t_idx] if t_idx is not None else None
            heading = self._headings[h_pos] if h_pos is not None else None
            # A row with both t_idx and h_pos is only "assigned" if the scorer
            # actually paired them; otherwise it is purely a display co-location.
            is_assigned = (
                t_idx is not None
                and heading is not None
                and self._track_assignments.get(t_idx) == heading
            )
            item = QTreeWidgetItem()

            item.setData(0, _ROLE_TRACK_IDX,   t_idx)
            item.setData(0, _ROLE_HEADING_POS, h_pos)
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if h_pos is not None:
                flags |= (Qt.ItemFlag.ItemIsDragEnabled
                          | Qt.ItemFlag.ItemIsDropEnabled
                          | Qt.ItemFlag.ItemIsEditable)
            item.setFlags(flags)

            item.setText(_COL_TRACK, _fmt_track(chapter) if chapter is not None else "")

            if is_assigned:
                item.setText(_COL_ASSIGNED, heading or "")
                item.setText(_COL_PENDING,  "")
            elif heading is not None:
                item.setText(_COL_ASSIGNED, "")
                item.setText(_COL_PENDING,  heading)
            else:
                item.setText(_COL_ASSIGNED, "")
                item.setText(_COL_PENDING,  "")

            self._tree.addTopLevelItem(item)

            if t_idx is not None and t_idx in out_of_order:
                tooltip = oo_tooltips.get(t_idx, "")
                for col in range(self._tree.columnCount()):
                    item.setBackground(col, yellow)
                    item.setForeground(col, black)
                    if tooltip:
                        item.setToolTip(col, tooltip)

            if chapter is not None:
                btn = self._make_play_btn(chapter)
                self._tree.setItemWidget(item, _COL_PLAY, btn)
                self._play_buttons[chapter.start_ms] = btn
                if chapter.start_ms == active_start_ms:
                    btn.setText("■")
                    self._active_play_btn = btn

            if is_assigned and t_idx is not None:
                arrow = QPushButton("→")
                arrow.setFixedSize(28, 22)
                arrow.setToolTip("Unassign")
                arrow.clicked.connect(
                    lambda checked=False, ti=t_idx: self._unassign_heading(ti)
                )
                self._tree.setItemWidget(item, _COL_ARROW, arrow)
            elif t_idx is not None and h_pos is not None:
                arrow = QPushButton("←")
                arrow.setFixedSize(28, 22)
                arrow.setToolTip("Assign")
                arrow.clicked.connect(
                    lambda checked=False, ti=t_idx, hp=h_pos: self._assign_collocated(ti, hp)
                )
                self._tree.setItemWidget(item, _COL_ARROW, arrow)

        if self._active_play_btn is not None:
            self._playback_timer.start()

        self._tree.verticalScrollBar().setValue(scroll_pos)
        self._rebuilding = False

    # ── File import ─────────────────────────────────────────────────────────

    def _import_m4b(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select M4B file", "", "M4B Audiobook (*.m4b);;All files (*)"
        )
        if not path:
            return
        self._m4b_path = Path(path)
        try:
            self._chapters = extract_chapters(self._m4b_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error reading M4B", str(exc))
            return
        self._m4b_label.setText(self._m4b_path.name)
        self._m4b_label.setStyleSheet("")
        self._refresh_tree()
        self._update_assign_btn()

    def _import_epub(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select EPUB file", "", "EPUB (*.epub);;All files (*)"
        )
        if not path:
            return
        self._epub_path = Path(path)
        try:
            self._headings = extract_toc(self._epub_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error reading EPUB", str(exc))
            return
        self._epub_label.setText(self._epub_path.name)
        self._epub_label.setStyleSheet("")
        self._refresh_tree()
        self._update_assign_btn()

    def _refresh_tree(self) -> None:
        self._stop_playback()
        self._track_assignments = {}
        self._claimed_headings = set()
        self._play_buttons = {}
        self._rebuild_view()

    # ── Assign button state ──────────────────────────────────────────────────

    def _update_assign_btn(self) -> None:
        ready = bool(self._m4b_path and self._epub_path and self._chapters and self._headings)
        self._assign_btn.setEnabled(ready)
        self._export_btn.setEnabled(bool(self._m4b_path and self._chapters))

    # ── Assignment ───────────────────────────────────────────────────────────

    def _start_assign(self) -> None:
        self._assign_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._model_combo.setEnabled(False)
        self._seconds_spin.setEnabled(False)
        self._progress.setValue(0)
        self._status_label.setText("Loading model…")

        self._stop_playback()
        self._track_assignments = {}
        self._claimed_headings = set()
        self._rebuild_view()

        model_name = self._model_combo.currentText()
        seconds = self._seconds_spin.value()
        self._worker = _Worker(
            self._chapters, self._m4b_path, self._headings, model_name, seconds
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.assignments_updated.connect(self._on_assignments_updated)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_assign_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_assignments_updated(self, assignments: dict) -> None:
        self._track_assignments = {k: v for k, v in assignments.items() if v}
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setValue(int(done / total * 100))
        self._status_label.setText(f"Scoring track {done} / {total}…")

    def _on_assign_done(self) -> None:
        self._progress.setValue(100)
        n = sum(1 for v in self._track_assignments.values() if v)
        self._status_label.setText(f"Done — {n} track(s) assigned.")
        self._assign_btn.setEnabled(True)
        self._export_btn.setEnabled(bool(self._m4b_path and self._chapters))
        self._model_combo.setEnabled(True)
        self._seconds_spin.setEnabled(True)

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Assignment failed", message)
        self._status_label.setText("Error.")
        self._assign_btn.setEnabled(True)
        self._export_btn.setEnabled(bool(self._m4b_path and self._chapters))
        self._model_combo.setEnabled(True)
        self._seconds_spin.setEnabled(True)

    # ── Heading editing ──────────────────────────────────────────────────────

    def _on_drop(
        self,
        src_h_pos: int,
        src_t_idx,          # int | None
        dst_h_pos,          # int | None
        dst_t_idx,          # int | None
        col: int,
        drop_above: bool,
    ) -> None:
        heading_str = self._headings[src_h_pos]

        # 1. Reorder — move the heading in self._headings.
        if dst_h_pos is not None:
            self._headings.pop(src_h_pos)
            adj = dst_h_pos - (1 if dst_h_pos > src_h_pos else 0)
            self._headings.insert(adj if drop_above else adj + 1, heading_str)

        # 2. Assign — only if not already claimed by another track.
        if col == _COL_ASSIGNED and dst_t_idx is not None:
            if heading_str not in self._claimed_headings:
                self._track_assignments[dst_t_idx] = heading_str

        # 3. Unassign.
        if col == _COL_PENDING and src_t_idx is not None:
            self._track_assignments.pop(src_t_idx, None)

        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _on_item_changed(self, item: QTreeWidgetItem) -> None:
        if self._rebuilding:
            return
        h_pos = item.data(0, _ROLE_HEADING_POS)
        if h_pos is None or h_pos >= len(self._headings):
            return
        old_name = self._headings[h_pos]
        new_name = (item.text(_COL_ASSIGNED) or item.text(_COL_PENDING)).strip()
        if not new_name:
            QTimer.singleShot(0, self._rebuild_view)
            return
        if new_name == old_name:
            return
        was_representative = (
            next((i for i, h in enumerate(self._headings) if h == old_name), None)
            == h_pos
        )
        self._headings[h_pos] = new_name
        if was_representative:
            for ti in list(self._track_assignments):
                if self._track_assignments[ti] == old_name:
                    self._track_assignments[ti] = new_name
        self._claimed_headings = set(self._track_assignments.values())
        QTimer.singleShot(0, self._rebuild_view)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        h_pos = item.data(0, _ROLE_HEADING_POS)
        t_idx = item.data(0, _ROLE_TRACK_IDX)

        is_assigned = (
            t_idx is not None
            and h_pos is not None
            and self._track_assignments.get(t_idx) == self._headings[h_pos]
        )

        menu = QMenu(self)
        if h_pos is not None:
            menu.addAction("Rename",    lambda: self._rename_heading(h_pos))
            menu.addAction("Add Above", lambda: self._add_heading(h_pos))
            menu.addAction("Add Below", lambda: self._add_heading(h_pos + 1))
            if h_pos > 0:
                menu.addAction("Combine with Above", lambda: self._combine_with_above(h_pos))
            menu.addAction("Delete",    lambda: self._delete_heading(h_pos))
            if t_idx is not None:
                menu.addSeparator()
                if is_assigned:
                    menu.addAction("Unassign", lambda: self._unassign_heading(t_idx))
                else:
                    menu.addAction("Assign to This Track",
                                   lambda: self._assign_collocated(t_idx, h_pos))
        elif t_idx is not None:
            menu.addAction("Assign…", lambda: self._show_heading_picker(t_idx))

        if not menu.isEmpty():
            menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _rename_heading(self, h_pos: int) -> None:
        old_name = self._headings[h_pos]
        new_name, ok = QInputDialog.getText(
            self, "Rename Heading", "New name:", text=old_name
        )
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        was_representative = (
            next((i for i, h in enumerate(self._headings) if h == old_name), None)
            == h_pos
        )
        self._headings[h_pos] = new_name
        if was_representative:
            for ti in list(self._track_assignments):
                if self._track_assignments[ti] == old_name:
                    self._track_assignments[ti] = new_name
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _add_heading(self, h_pos: int) -> None:
        name, ok = QInputDialog.getText(self, "Add Heading", "Heading name:")
        if not ok or not name.strip():
            return
        self._headings.insert(h_pos, name.strip())
        self._rebuild_view()

    def _delete_heading(self, h_pos: int) -> None:
        old_name = self._headings.pop(h_pos)
        if old_name not in self._headings:
            self._track_assignments = {
                k: v for k, v in self._track_assignments.items() if v != old_name
            }
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _combine_with_above(self, h_pos: int) -> None:
        above_name = self._headings[h_pos - 1]
        current_name = self._headings[h_pos]
        combined = f"{above_name} / {current_name}"
        self._headings[h_pos - 1] = combined
        self._headings.pop(h_pos)
        for ti in list(self._track_assignments):
            if self._track_assignments[ti] in (above_name, current_name):
                self._track_assignments[ti] = combined
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _export(self) -> None:
        if not self._m4b_path or not self._chapters:
            return
        m4b_path: Path = self._m4b_path

        from .extractor import get_duration_ms
        from .types import AlignedChapter
        from .writer import write_chapters

        unassigned_count = sum(
            1 for i in range(len(self._chapters))
            if not self._track_assignments.get(i)
        )
        if unassigned_count:
            reply = QMessageBox.question(
                self,
                "Unassigned Tracks",
                f"{unassigned_count} track(s) have no assigned heading. "
                "Export anyway? (Unassigned tracks will keep their original names.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        default_path = m4b_path.with_stem(m4b_path.stem + "_chaptered")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export M4B",
            str(default_path),
            "M4B Audiobook (*.m4b);;All files (*)",
        )
        if not out_path:
            return

        aligned = [
            AlignedChapter(
                title=self._track_assignments.get(i) or self._chapters[i].title,
                start_ms=self._chapters[i].start_ms,
            )
            for i in range(len(self._chapters))
        ]
        try:
            total_ms = get_duration_ms(m4b_path)
            write_chapters(m4b_path, Path(out_path), aligned, total_ms)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{out_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _assign_collocated(self, track_idx: int, h_pos: int) -> None:
        """Assign the co-located (display-only) heading to its adjacent track."""
        heading = self._headings[h_pos]
        if heading in self._claimed_headings:
            QMessageBox.warning(
                self, "Already Assigned",
                f'"{heading}" is already assigned to another track.',
            )
            return
        self._track_assignments[track_idx] = heading
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _unassign_heading(self, track_idx: int) -> None:
        self._track_assignments.pop(track_idx, None)
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def _show_heading_picker(self, track_idx: int) -> None:
        unassigned = [h for h in self._headings if h not in self._claimed_headings]
        if not unassigned:
            QMessageBox.information(
                self, "No Headings Available", "All headings are already assigned."
            )
            return
        heading, ok = QInputDialog.getItem(
            self,
            "Assign Heading",
            f"Select heading for Track {track_idx + 1}:",
            unassigned,
            0,
            False,
        )
        if not ok or not heading:
            return
        self._track_assignments[track_idx] = heading
        self._claimed_headings = set(self._track_assignments.values())
        self._rebuild_view()

    def closeEvent(self, event) -> None:
        self._stop_playback()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
