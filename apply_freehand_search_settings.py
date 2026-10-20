#!/usr/bin/env python3
"""Apply the Freehand Search & Indexing settings-page integration to main.py.

Run this file in Freehand's root folder alongside freehand_search_index.py.
Creates a backup and compiles the modified main.py before replacing it.
"""
from __future__ import annotations
import argparse
import ast
import os
from pathlib import Path

SEARCH_METHODS = '''    # ----- Freehand local Search & Indexing settings -------------------------
    def _search_settings_visible_roots(self):
        roots = self.search_index.get_scopes()
        maximum = max(0, (len(roots)-1)//6*6)
        self.search_settings_offset = min(maximum, max(0, self.search_settings_offset))
        return roots[self.search_settings_offset:self.search_settings_offset+6]

    def _search_settings_browse(self):
        # Non-native Qt modal chooser keeps the dialog above the glass overlay,
        # including Windows systems where native dialogs sometimes open behind it.
        from PySide6.QtWidgets import QFileDialog
        dialog = QFileDialog(self, "Add a folder to Freehand search", str(Path.home()))
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        if dialog.exec() == QDialog.Accepted:
            paths = dialog.selectedFiles()
            if paths:
                try:
                    added = self.search_index.add_root(paths[0])
                    self.search_settings_offset = max(0, (len(self.search_index.get_scopes())-1)//6*6)
                    self.status = ("Added folder to search index: " if added else "Folder already indexed: ") + paths[0]
                except (OSError, ValueError) as exc:
                    self.status = "Cannot index folder: " + str(exc)
        self.update()

    def _search_settings_click(self, pressed):
        if pressed("search_add"):
            self._search_settings_browse()
        elif pressed("search_auto"):
            auto = not self.search_index.automatic()
            self.search_index.set_automatic(auto)
            self.status = "Automatic indexing " + ("enabled" if auto else "disabled")
        elif pressed("search_rebuild"):
            self.search_index.request_scan()
            self.status = "File index refresh requested"
        elif pressed("search_prev"):
            self.search_settings_offset = max(0, self.search_settings_offset-6)
        elif pressed("search_next"):
            count = len(self.search_index.get_scopes())
            self.search_settings_offset = min(max(0, (count-1)//6*6), self.search_settings_offset+6)
        else:
            visible = self._search_settings_visible_roots()
            for i, root in enumerate(visible):
                try:
                    if pressed(f"search_toggle_{i}"):
                        self.search_index.set_enabled(root["path"], not root["enabled"])
                        self.status = "Updated indexed folder: " + root["path"]
                        break
                    if pressed(f"search_remove_{i}"):
                        if os.path.normcase(root["path"]) == os.path.normcase(str(Path.home())):
                            self.status = "Home can be switched off, but cannot be removed."
                        else:
                            self.search_index.remove_root(root["path"])
                            self.status = "Removed indexed folder: " + root["path"]
                        break
                except (OSError, ValueError) as exc:
                    self.status = "Index settings: " + str(exc)
        self.update()

    def _paint_search_settings_page(self, painter, controls, text_rects, scale):
        def button(key, label, active=False, muted=False):
            rect = controls[key]
            fill = QColor(53, 139, 82, 230) if active else QColor(68, 76, 89, 210)
            if muted:
                fill = QColor(35, 51, 65, 155)
            edge = QColor(153, 241, 181, 220) if active else QColor(155, 168, 182, 185)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(edge, 1.5*scale))
            painter.drawRoundedRect(rect, 9*scale, 9*scale)
            painter.setPen(QColor(233, 249, 253) if not muted else QColor(142, 166, 177))
            self._draw_fitted_button_label(painter, rect, label,
                                           base_size=max(8, int(9*scale)), min_size=7, inset=7*scale)

        button("search_add", "+ BROWSE / ADD FOLDER")
        button("search_rebuild", "SCAN / REBUILD INDEX")
        button("search_auto", "AUTO UPDATE  ·  " + ("ON" if self.search_index.automatic() else "OFF"),
               active=self.search_index.automatic())
        count = len(self.search_index.get_scopes())
        button("search_prev", "◀ PREVIOUS", muted=self.search_settings_offset <= 0)
        button("search_next", "NEXT ▶", muted=self.search_settings_offset+6 >= count)
        painter.setPen(QColor(185, 226, 239))
        painter.setFont(self._ui_font(8, QFont.DemiBold))
        visible = self._search_settings_visible_roots()
        for i in range(6):
            key = f"search_row_{i}"
            if i >= len(visible):
                continue
            root = visible[i]
            directory = root["path"]
            home = os.path.normcase(directory) == os.path.normcase(str(Path.home()))
            rect = controls[key]
            painter.setBrush(QBrush(QColor(8, 35, 53, 192)))
            painter.setPen(QPen(QColor(109, 174, 193, 170), 1*scale))
            painter.drawRoundedRect(rect, 9*scale, 9*scale)
            painter.setFont(self._ui_font(8, QFont.Normal))
            painter.setPen(QColor(229, 250, 253))
            caption = ("HOME  ·  " if home else "FOLDER  ·  ") + directory
            label = painter.fontMetrics().elidedText(caption, Qt.ElideMiddle, max(20, int(rect.width()-16*scale)))
            painter.drawText(rect.adjusted(8*scale, 0, -8*scale, 0), Qt.AlignLeft | Qt.AlignVCenter, label)
            button(f"search_toggle_{i}", "ON" if root["enabled"] else "OFF", active=root["enabled"])
            button(f"search_remove_{i}", "DEFAULT" if home else "REMOVE", muted=home)
        st = self.search_index.status()
        summary = (f"{st['count']:,} indexed items  ·  "
                   + (f"Scanning {st['scanned']:,}..." if st['scanning'] else "Ready")
                   + f"  ·  {count} locations")
        if st['error']:
            summary += "  ·  " + st['error'][:70]
        painter.setPen(QColor(189, 233, 239))
        self._draw_wrapped_fitted_text(painter, text_rects["search_info"],
                                       summary + "\\nLocal only. Subfolders included. Home is indexed by default.",
                                       base_size=max(8, int(9*scale)), min_size=7,
                                       align=Qt.AlignLeft | Qt.AlignTop)

'''

SEARCH_LAYOUT = '''        elif page == "search":
            # Full-width, paginated root list shared by painting and hit testing.
            buttons["search_add"] = (L, 0.0, W, 40.0)
            buttons["search_rebuild"] = (R, 0.0, W, 40.0)
            buttons["search_auto"] = (L, 50.0, W, 40.0)
            buttons["search_prev"] = (R, 50.0, 145.0, 40.0)
            buttons["search_next"] = (R+155.0, 50.0, 145.0, 40.0)
            for i in range(6):
                row_y = 108.0+i*44.0
                buttons[f"search_row_{i}"] = (L, row_y, 420.0, 37.0)
                buttons[f"search_toggle_{i}"] = (L+428.0, row_y, 100.0, 37.0)
                buttons[f"search_remove_{i}"] = (L+536.0, row_y, 104.0, 37.0)
            texts["search_info"] = (L, 387.0, 640.0, 44.0)
            y = 440.0
            right_h = y
'''


def apply(source):
    original = source
    def replace(old, new, label):
        nonlocal source
        hits = source.count(old)
        if hits != 1:
            raise RuntimeError(f'{label}: expected exactly one anchor, found {hits}')
        source = source.replace(old, new, 1)

    if 'def _paint_search_settings_page(' in source:
        raise RuntimeError('Search settings integration already installed; no changes made.')
    replace('from search_launcher import SearchLauncherDialog\n',
            'from search_launcher import SearchLauncherDialog\nfrom freehand_search_index import SearchIndex\n', 'SearchIndex import')
    replace('        self.settings_open = False\n        self.settings_page = "interaction"\n',
            '        self.settings_open = False\n        self.settings_page = "interaction"\n'
            '        self.search_index = SearchIndex()\n        self.search_index.start()\n'
            '        self.search_settings_offset = 0\n', 'instance initialization')
    replace('    def _settings_layout(self, page=None):\n',
            SEARCH_METHODS+'    def _settings_layout(self, page=None):\n', 'search settings methods')
    replace('        elif page == "virtual":\n            stack("virtual_scene_switch")',
            SEARCH_LAYOUT+'        elif page == "virtual":\n            stack("virtual_scene_switch")', 'page geometry')
    replace('        widths = [78.0, 51.0, 42.0, 53.0, 69.0, 47.0, 68.0, 54.0, 76.0, 47.0]',
            '        widths = [74.0, 49.0, 42.0, 53.0, 66.0, 46.0, 65.0, 48.0, 69.0, 62.0, 47.0]',
            'tab widths')
    replace('        total = sum(widths)*scale + tab_gap*9.0',
            '        total = sum(widths)*scale + tab_gap*(len(widths)-1)', 'tab width summation')
    replace('        names = ("interaction", "hands", "tli", "notifications", "gestures", "voice", "d1", "sounds", "virtual", "update")',
            '        names = ("interaction", "hands", "tli", "notifications", "gestures", "voice", "d1", "sounds", "virtual", "search", "update")',
            'tab names')
    replace('                              "d1": "ENGINE", "update": "UPDATE", "virtual": "VIRTUAL", "sounds": "SOUNDS"}',
            '                              "d1": "ENGINE", "update": "UPDATE", "virtual": "VIRTUAL", "sounds": "SOUNDS", "search": "SEARCH"}',
            'tab captions')
    replace('                                            "notifications", "voice", "update", "d1", "virtual", "sounds"):',
            '                                            "notifications", "voice", "update", "d1", "virtual", "sounds", "search"):',
            'paint page allowlist')
    replace('                    elif self.settings_page == "d1":\n                        info = d1_status()',
            '                    elif self.settings_page == "search":\n'
            '                        self._paint_search_settings_page(painter, controls, text_rects, scale)\n'
            '                    elif self.settings_page == "d1":\n                        info = d1_status()',
            'search paint')
    replace('            if self.settings_page == "d1":\n                if pressed("d1_log_up"):',
            '            if self.settings_page == "search":\n'
            '                self._search_settings_click(pressed)\n                return\n'
            '            if self.settings_page == "d1":\n                if pressed("d1_log_up"):',
            'search click')
    replace('    def closeEvent(self, event):\n        if self._wake_word_save_timer.isActive():',
            '    def closeEvent(self, event):\n        self.search_index.close()\n'
            '        if self._wake_word_save_timer.isActive():', 'clean shutdown')
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', default='main.py', help='Path to the Freehand main.py to modify')
    args = parser.parse_args()
    path = Path(args.main).resolve()
    if not (path.parent / 'freehand_search_index.py').is_file():
        parser.error('Put freehand_search_index.py in the same directory as main.py.')
    if not path.is_file():
        parser.error('Cannot find main.py. Run this script in your Freehand folder.')
    text = path.read_text(encoding='utf-8')
    if 'def _paint_search_settings_page(' in text:
        required = (
            'from freehand_search_index import SearchIndex',
            'self.search_index = SearchIndex()',
            'self.search_index.start()',
            'self.search_index.close()',
            '"search": "SEARCH"',
        )
        if not all(marker in text for marker in required):
            parser.exit(2, 'ERROR: main.py has an incomplete Search integration; no changes made.\n')
        compile(text, str(path), 'exec')
        print('Search & Indexing already integrated; main.py unchanged.')
        return
    try:
        revised = apply(text)
    except Exception as exc:
        parser.exit(2, 'No files changed: '+str(exc)+'\n')
    backup = path.with_name(path.name+'.before-search-settings.bak')
    if backup.exists():
        parser.error(f'Backup already exists: {backup}; move it before patching another copy.')
    backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix('.search-settings.tmp')
    try:
        temporary.write_text(revised, encoding='utf-8')
        compile(revised, str(path), 'exec')
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        backup.replace(path)
        raise
    print('Search tab integrated successfully:', path)
    print('Original backed up to:', backup)
    print('The indexing database will be created in the user data directory at next launch.')

if __name__ == '__main__':
    main()
