"""
player_panel.py — left-side player browser.

Classes:
    PlayerItemWidget  — compact list-row widget with status indicators
    PlayerListPanel   — searchable/sortable player list with signals
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QLineEdit, QFrame, QSizePolicy, QToolButton, QMenu,
)

from constants import _parse_player_id


# ── PlayerItemWidget ───────────────────────────────────────────────────────────

class PlayerItemWidget(QWidget):
    """Compact widget displayed inside each QListWidget item."""

    def __init__(self, player_id: str,
                 has_video: bool, has_features: bool, has_anomaly: bool,
                 n_frames: int = 0):
        super().__init__()
        year, name, pos = _parse_player_id(player_id)
        self.player_id = player_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(1)

        # Top row: name + year
        top = QHBoxLayout()
        top.setSpacing(4)
        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(
            "font-weight:bold; font-size:12px; color:#e0e0e0;" if has_video
            else "font-size:12px; color:#555;")
        top.addWidget(lbl_name)
        top.addStretch()
        lbl_year = QLabel(str(year) if year else "")
        lbl_year.setStyleSheet("color:#777; font-size:12px;")
        top.addWidget(lbl_year)
        layout.addLayout(top)

        # Bottom row: position + status dots
        bot = QHBoxLayout()
        bot.setSpacing(6)
        lbl_pos = QLabel(pos)
        lbl_pos.setStyleSheet("color:#666; font-size:12px;")
        bot.addWidget(lbl_pos)
        if n_frames:
            lbl_frames = QLabel(f"{n_frames} fr")
            lbl_frames.setStyleSheet("color:#555; font-size:12px;")
            bot.addWidget(lbl_frames)
        bot.addStretch()
        for letter, available, tip in [
            ("V", has_video,    "Video file found"),
            ("F", has_features, "Features CSV entry found"),
            ("A", has_anomaly,  "Anomaly data found"),
        ]:
            dot = QLabel(f"<b>{letter}</b>")
            dot.setFixedWidth(14)
            dot.setAlignment(Qt.AlignCenter)
            if available:
                dot.setStyleSheet("color:#69f0ae; font-size:12px;")
            else:
                dot.setStyleSheet("color:#444; font-size:12px;")
            dot.setToolTip(tip if available else f"{tip[:-5]} not found")
            bot.addWidget(dot)
        layout.addLayout(bot)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if not has_video:
            self.setToolTip("No .mp4 video file found for this player")


# ── PlayerListPanel ────────────────────────────────────────────────────────────

class PlayerListPanel(QWidget):
    """Left panel: searchable player browser with status indicators."""

    player_selected         = pyqtSignal(str)   # emits player_id
    request_features_folder = pyqtSignal()
    request_anomaly_folder  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        self._lbl_count = QLabel("Players")
        self._lbl_count.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold;")
        hdr.addWidget(self._lbl_count)
        hdr.addStretch()

        sort_btn = QToolButton()
        sort_btn.setText("Sort ▾")
        sort_btn.setPopupMode(QToolButton.InstantPopup)
        sort_btn.setToolTip("Sort player list")
        sort_btn.setStyleSheet("font-size:12px; color:#888;")
        sort_menu = QMenu(sort_btn)
        sort_menu.addAction("By Year",         lambda: self._sort("year"))
        sort_menu.addAction("By Name (A-Z)",   lambda: self._sort("name"))
        sort_menu.addAction("By Name (Z-A)",   lambda: self._sort("name_rev"))
        sort_menu.addAction("By Status",       lambda: self._sort("status"))
        sort_btn.setMenu(sort_menu)
        hdr.addWidget(sort_btn)
        layout.addLayout(hdr)

        # Search box
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter players…")
        self._search.setToolTip("Type to filter players by name, year, or position")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # List
        self._list = QListWidget()
        self._list.setSpacing(1)
        self._list.setStyleSheet(
            "QListWidget { border:none; background:#1e1e1e; }"
            "QListWidget::item { border-bottom: 1px solid #2a2a2a; }"
            "QListWidget::item:selected { background:#1e3a5f; border:1px solid #2a5a8f; }"
            "QListWidget::item:hover:!selected { background:#252525; }"
        )
        self._list.setToolTip("Click a player to view their pose data.\n"
                              "Greyed-out players have no video file.")
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#333;")
        layout.addWidget(sep)

        btn_feat = QPushButton("+ Features Folder…")
        btn_feat.setToolTip("Select a folder containing a feature CSV to load\n"
                            "aggregate statistics for each player")
        btn_feat.setStyleSheet("font-size:12px; color:#888; background:#252525;")
        btn_feat.clicked.connect(self.request_features_folder)
        layout.addWidget(btn_feat)

        btn_anom = QPushButton("+ Anomaly Folder…")
        btn_anom.setToolTip("Select a folder containing an anomaly CSV\n"
                            "to show STG-NF flagged frames on the timeline")
        btn_anom.setStyleSheet("font-size:12px; color:#888; background:#252525;")
        btn_anom.clicked.connect(self.request_anomaly_folder)
        layout.addWidget(btn_anom)

        self._player_data: list[dict] = []
        self._current_pid: str | None = None

    def populate(self, player_data: list[dict]):
        self._player_data = player_data
        self._rebuild_list()

    def _rebuild_list(self):
        self._list.blockSignals(True)
        prev_pid = self._current_pid
        self._list.clear()
        for info in self._player_data:
            pid = info["player_id"]
            item = QListWidgetItem()
            widget = PlayerItemWidget(
                pid,
                info.get("has_video", False),
                info.get("has_features", False),
                info.get("has_anomaly", False),
                info.get("n_frames", 0),
            )
            item.setSizeHint(QSize(220, 52))
            item.setData(Qt.UserRole, pid)
            if not info.get("has_video", False):
                item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
        self._list.blockSignals(False)
        self._apply_filter(self._search.text())
        self._lbl_count.setText(f"Players ({len(self._player_data)})")
        if prev_pid:
            self.select_player(prev_pid)

    def _sort(self, key: str):
        def sort_key(info):
            year, name, pos = _parse_player_id(info["player_id"])
            status = (info.get("has_video", False), info.get("has_features", False))
            if key == "year":
                return (year, name)
            if key == "name":
                return name
            if key == "name_rev":
                return name
            if key == "status":
                return (not status[0], not status[1], name)
            return name
        reverse = key == "name_rev"
        self._player_data.sort(key=sort_key, reverse=reverse)
        self._rebuild_list()

    def _apply_filter(self, text: str):
        text = text.lower().strip()
        for i in range(self._list.count()):
            item = self._list.item(i)
            pid = item.data(Qt.UserRole) or ""
            item.setHidden(bool(text) and text not in pid.lower())

    def _on_item_changed(self, current, _previous):
        if current is None:
            return
        pid = current.data(Qt.UserRole)
        if pid and pid != self._current_pid:
            self._current_pid = pid
            self.player_selected.emit(pid)

    def select_player(self, pid: str):
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == pid:
                self._list.setCurrentItem(item)
                self._list.scrollToItem(item)
                self._current_pid = pid
                break

    def refresh_status(self, player_data: list[dict]):
        """Update has_features / has_anomaly flags without full rebuild."""
        pid_map = {d["player_id"]: d for d in player_data}
        self._player_data = [pid_map.get(d["player_id"], d) for d in self._player_data]
        self._rebuild_list()
