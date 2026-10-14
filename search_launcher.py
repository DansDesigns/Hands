"""Ulauncher-style local search window for Spatial Desktop (no AI model required)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize, QUrl, QRectF
from PySide6.QtGui import QDesktopServices, QFont, QKeySequence, QShortcut, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QFileDialog, QMessageBox, QApplication, QAbstractItemView, QWidget,
)

from local_file_search import LocalFileIndex, SearchResult


class _IndexWorker(QThread):
    progress = Signal(str)
    completed = Signal(int, int)
    failed = Signal(str)

    def __init__(self, index: LocalFileIndex, roots: list[str]):
        super().__init__()
        self.index = index
        self.roots = roots

    def run(self):
        try:
            seen, changed = self.index.index_roots(self.roots, self.progress.emit, self.isInterruptionRequested)
            self.completed.emit(seen, changed)
        except Exception as exc:
            self.failed.emit(str(exc))


class SearchLauncherDialog(QDialog):
    """A small keyboard-first launcher with live local file ranking and Vosk dictation."""
    dictation_toggled = Signal(bool)
    speak_requested = Signal(str)
    closed = Signal()

    def __init__(self, index: LocalFileIndex | None = None, parent=None):
        super().__init__(parent)
        self.index = index or LocalFileIndex()
        self.dictation_enabled = False
        self.index_worker: _IndexWorker | None = None
        self._index_initialised = False
        self._last_results: list[SearchResult] = []
        self.setWindowTitle("Spatial Search")
        # Match the glass menu: no native title bar or window decorations.
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # Closing the launcher must never make QApplication quit: the fullscreen
        # Spatial Desktop overlay is a Qt.Tool window and may not count as a normal
        # top-level window for last-window-close purposes.
        self.setAttribute(Qt.WA_QuitOnClose, False)
        # The window itself is transparent and paintEvent draws the rounded glass
        # panel, so all four corners are rounded on every OS/compositor (an opaque
        # frameless window keeps square corners behind a stylesheet radius).
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._details_expanded = False
        self._user_collapsed_details = False
        self._compact_height = 78
        self._expanded_height = 470
        self.resize(680, self._compact_height)
        self.setMinimumSize(480, 68)
        self.setStyleSheet("""
            QDialog { background:transparent; color:#e8f7ff; border:none; }
            QLabel { color:#a9c6d8; }
            QLineEdit { background:#111f31; color:#f3fbff; border:1px solid #47758e;
                        border-radius:12px; padding:7px 12px; font-size:17px; selection-background-color:#267a9c; }
            QListWidget { background:#0e1928; color:#eaf7ff; border:1px solid #233e52; border-radius:12px;
                          padding:6px; outline:0; font-size:13px; }
            QListWidget::item { padding:9px 10px; border-radius:8px; margin:2px 0; }
            QListWidget::item:selected { background:#17465d; border:1px solid #6bd3e6; }
            QPushButton { background:#162c40; color:#e5f6ff; border:1px solid #355b74; border-radius:9px; padding:8px 12px; }
            QPushButton:hover { background:#20465d; border-color:#62cce1; }
            QPushButton:checked { background:#1e684e; border-color:#58dba9; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Compact launcher row. The expanded result list and utility buttons live
        # in a collapsible section below, keeping the default UI visually quiet.
        top = QHBoxLayout()
        top.setSpacing(8)
        mark = QLabel("⌕")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(30, 38)
        mark.setStyleSheet("font-size:25px; color:#7ed8e9; font-weight:600;")
        top.addWidget(mark)

        self.query = QLineEdit()
        self.query.setPlaceholderText("Search files, folders and document contents…")
        self.query.setClearButtonEnabled(True)
        self.query.setFont(QFont("Segoe UI", 15))
        self.query.setFixedHeight(42)
        top.addWidget(self.query, 1)

        self.details_toggle = QPushButton("▾")
        self.details_toggle.setFixedSize(42, 42)
        self.details_toggle.setToolTip("Show search results and tools")
        self.details_toggle.clicked.connect(self.toggle_details)
        top.addWidget(self.details_toggle)

        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(38, 42)
        self.close_button.setToolTip("Close search (Spatial Desktop stays running)")
        self.close_button.setStyleSheet("QPushButton { font-size:20px; padding:0; }")
        self.close_button.clicked.connect(self.close)
        top.addWidget(self.close_button)
        layout.addLayout(top)

        self.details = QWidget(self)
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)

        self.results = QListWidget()
        self.results.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results.setUniformItemSizes(False)
        self.results.itemActivated.connect(lambda _item: self.open_selected())
        self.results.setMinimumHeight(160)
        details_layout.addWidget(self.results, 1)

        footer = QHBoxLayout()
        self.status = QLabel("Starting local file index…")
        self.status.setMinimumWidth(150)
        self.status.setStyleSheet("font-size:11px;")
        footer.addWidget(self.status, 1)
        self.dictate_button = QPushButton("🎙 Dictate")
        self.dictate_button.setCheckable(True)
        self.dictate_button.setToolTip("Stream partial words from local Vosk speech recognition")
        self.dictate_button.toggled.connect(self._on_dictation_button)
        footer.addWidget(self.dictate_button)
        add_folder = QPushButton("Add Folder")
        add_folder.clicked.connect(self._add_folder)
        footer.addWidget(add_folder)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_index)
        footer.addWidget(refresh)
        speak_button = QPushButton("Speak")
        speak_button.setToolTip("Speak the selected file name and matching excerpt with local Kokoro TTS")
        speak_button.clicked.connect(self.speak_selected)
        footer.addWidget(speak_button)
        open_button = QPushButton("Open")
        open_button.clicked.connect(self.open_selected)
        footer.addWidget(open_button)
        details_layout.addLayout(footer)
        layout.addWidget(self.details, 1)
        self.details.hide()

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(85)
        self._search_timer.timeout.connect(self._run_search)
        self.query.textChanged.connect(self._on_query_changed)
        self.query.textEdited.connect(self._on_manual_edit)

        self._toggle_shortcut = QShortcut(QKeySequence("Ctrl+Space"), self)
        self._toggle_shortcut.activated.connect(self.close)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#3e8198"), 1.0))
        painter.setBrush(QColor("#0b1320"))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)
        painter.end()

    def _set_details_expanded(self, expanded: bool, user_action: bool = False):
        self._details_expanded = bool(expanded)
        if user_action:
            self._user_collapsed_details = not self._details_expanded
        self.details.setVisible(self._details_expanded)
        self.details_toggle.setText("▴" if self._details_expanded else "▾")
        self.details_toggle.setToolTip("Hide search results and tools" if self._details_expanded
                                       else "Show search results and tools")
        self.resize(self.width(), self._expanded_height if self._details_expanded else self._compact_height)

    def toggle_details(self):
        self._set_details_expanded(not self._details_expanded, user_action=True)

    def _on_query_changed(self, *_args):
        self._schedule_search()
        # Keep first launch compact, but reveal results automatically once there
        # is a real query. The arrow can collapse them again at any time.
        if len(self.query.text().strip()) >= 2 and not self._details_expanded and not self._user_collapsed_details:
            self._set_details_expanded(True)

    def showEvent(self, event):
        super().showEvent(event)
        self.query.setFocus(Qt.ActiveWindowFocusReason)
        self.query.selectAll() if self.query.text() else None
        QTimer.singleShot(0, self._initialise_index)

    def _initialise_index(self):
        if self.isVisible():
            if not self._index_initialised:
                self._index_initialised = True
                if self.index.count() == 0:
                    self.status.setText("Building your local file index…")
                else:
                    self.status.setText(f"{self.index.count():,} files indexed · updating in background")
                self.refresh_index()
            else:
                self.status.setText(f"{self.index.count():,} files indexed · local search ready")
            self._schedule_search()

    def set_dictation_enabled(self, enabled: bool, notify: bool = False):
        self.dictation_enabled = bool(enabled)
        self.dictate_button.blockSignals(True)
        self.dictate_button.setChecked(self.dictation_enabled)
        self.dictate_button.setText("🎙 Dictating" if self.dictation_enabled else "🎙 Dictate")
        self.dictate_button.blockSignals(False)
        if notify:
            self.dictation_toggled.emit(self.dictation_enabled)

    def set_dictation_text(self, text: str, final: bool = False):
        if not self.dictation_enabled or not self.isVisible():
            return
        clean = " ".join(text.split())
        if clean and self.query.text() != clean:
            self.query.setText(clean)
            self.query.setCursorPosition(len(clean))
        elif final and not clean:
            return

    def _on_dictation_button(self, enabled: bool):
        self.set_dictation_enabled(enabled)
        self.dictation_toggled.emit(bool(enabled))
        if enabled:
            self.query.setPlaceholderText("Speak naturally — words appear as Vosk recognises them…")
            self.query.setFocus()
        else:
            self.query.setPlaceholderText("Search files, folders and document contents…")

    def _on_manual_edit(self, _text: str):
        # Once a person starts typing, stop dictation so a later audio partial
        # cannot unexpectedly overwrite their manual query edits.
        self._user_collapsed_details = False
        if self.dictation_enabled:
            self.set_dictation_enabled(False)
            self.dictation_toggled.emit(False)
        if len(self.query.text().strip()) >= 2 and not self._details_expanded:
            self._set_details_expanded(True)

    def _schedule_search(self, *_args):
        self._search_timer.start()

    def _run_search(self):
        text = self.query.text().strip()
        self.results.clear()
        if len(text) < 2:
            self.status.setText(f"{self.index.count():,} files indexed · type or dictate to search")
            return
        try:
            found = self.index.search(text, limit=16)
        except Exception as exc:
            self.status.setText(f"Search error: {exc}")
            return
        # Keep live typing responsive: local SQLite/FTS ranking is deliberately
        # used directly here. Running the optional d1 model on every keystroke
        # can stall the UI and consume substantial CPU even for simple queries.
        self._last_results = found
        for result in found:
            display = result.name + "\n" + result.parent
            if result.snippet:
                display += "\n" + result.snippet
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, result.path)
            item.setToolTip(result.path + ("\n\n" + result.snippet if result.snippet else ""))
            item.setSizeHint(QSize(700, 68 if result.snippet else 52))
            self.results.addItem(item)
        if found:
            self.results.setCurrentRow(0)
            self.status.setText(f"{len(found)} results · local search · {self.index.count():,} files indexed")
        else:
            self.status.setText(f"No matches yet · {self.index.count():,} files indexed")

    def refresh_index(self):
        if self.index_worker and self.index_worker.isRunning():
            self.status.setText("Index refresh already running…")
            return
        roots = self.index.roots()
        if not roots:
            self.status.setText("Add a folder to start searching")
            return
        self.index_worker = _IndexWorker(self.index, roots)
        self.index_worker.progress.connect(self._index_progress)
        self.index_worker.completed.connect(self._index_completed)
        self.index_worker.failed.connect(self._index_failed)
        self.index_worker.start()

    def _index_progress(self, message: str):
        self.status.setText(message[:180])
        if self.query.text().strip():
            self._schedule_search()

    def _index_completed(self, seen: int, changed: int):
        self.status.setText(f"Index ready · {self.index.count():,} files · {changed:,} new/updated")
        self._schedule_search()

    def _index_failed(self, message: str):
        self.status.setText("Index error: " + message[:140])

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add a folder to local search", str(Path.home()))
        if not folder:
            return
        added, message = self.index.add_root(folder)
        self.status.setText(message)
        if added:
            self.refresh_index()

    def current_path(self) -> str | None:
        item = self.results.currentItem()
        return str(item.data(Qt.UserRole)) if item else None

    def open_selected(self):
        path = self.current_path()
        if not path:
            if self.results.count():
                self.results.setCurrentRow(0)
                path = self.current_path()
        if not path:
            return
        try:
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            if not opened:
                self.status.setText("Could not open file: " + path)
            else:
                self.close()
        except Exception as exc:
            self.status.setText(f"Could not open file: {exc}")

    def speak_selected(self):
        path = self.current_path()
        if not path:
            if self.results.count():
                self.results.setCurrentRow(0)
                path = self.current_path()
        if not path:
            return
        item = self.results.currentItem()
        detail = item.text().splitlines() if item else []
        text = Path(path).name
        if len(detail) > 2:
            text += ". " + detail[-1]
        self.speak_requested.emit(text[:500])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        if event.key() in (Qt.Key_Down, Qt.Key_Up) and self.query.hasFocus() and self.results.count():
            delta = 1 if event.key() == Qt.Key_Down else -1
            row = self.results.currentRow()
            self.results.setCurrentRow(max(0, min(self.results.count() - 1, row + delta)))
            return
        if event.key() == Qt.Key_Return and self.query.hasFocus():
            if self.results.count():
                self.open_selected()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.set_dictation_enabled(False)
        self.dictation_toggled.emit(False)
        if self.index_worker and self.index_worker.isRunning():
            self.index_worker.requestInterruption()
            self.index_worker.wait(1200)
        self.closed.emit()
        super().closeEvent(event)
