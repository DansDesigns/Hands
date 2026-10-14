"""Regression tests for physical-screen pointer mapping with Qtile reserved bars."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")


def _method(signature, next_signature):
    start = SOURCE.index(signature)
    end = SOURCE.index(next_signature, start + len(signature))
    return SOURCE[start:end]


def test_linux_virtual_pointer_uses_screen_origin_not_work_area_origin():
    method = _method("    def _virtual_point_to_global(self, point):",
                     "    def _set_os_pointer(self, point):")
    assert 'sys.platform.startswith("linux")' in method
    assert "QApplication.primaryScreen()" in method
    assert "screen.geometry()" in method
    assert "geometry.topLeft() + pixel" in method


def test_cursor_setter_uses_shared_full_screen_coordinate_transform():
    method = _method("    def _set_os_pointer(self, point):",
                     "    def _release_fist_mouse(self, side):")
    assert "self._virtual_point_to_global(point)" in method
    assert "self.mapToGlobal(QPoint(round(point.x()), round(point.y())))" not in method


def test_camera_mapping_and_tli_clamp_use_full_physical_screen_dimensions():
    camera_map = _method("    def interactive_visual_point(self, x, y):",
                         "    def interactive_hand_anchor")
    tli_map = _method("    def _scale_hand_tli_point(self, cursor, points=None, side=None):",
                      "    def _visual_tli_position")
    assert "screen.geometry()" in camera_map
    assert "screen_width-1" in camera_map
    assert "screen_height-1" in camera_map
    assert "geometry.width()" in tli_map and "geometry.height()" in tli_map
    assert "self.width()-1.0" not in tli_map and "self.height()-1.0" not in tli_map


def test_linux_fullscreen_geometry_is_revalidated_during_scene_updates():
    method = _method("    def _maintain_native_topmost(self, now=None):",
                     "    def _sync_linux_fullscreen_geometry(self):")
    assert "_last_linux_geometry_sync_check" in method
    assert "self._sync_linux_fullscreen_geometry()" in method
