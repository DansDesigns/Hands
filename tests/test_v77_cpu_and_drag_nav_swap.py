"""Regression checks for drag-scroll Back/Forward inversion and CPU reductions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


def _method(source, signature, next_signature):
    start = source.index(signature)
    end = source.index(next_signature, start + len(signature))
    return source[start:end]


def test_drag_scroll_swap_setting_is_saved_and_defaults_off():
    assert '"swap_drag_scroll_notches": False' in MAIN
    assert 'swap_drag_scroll_notches={str(bool(values.get(\'swap_drag_scroll_notches\', False))).lower()}' in MAIN
    assert 'elif key in ("invert_scroll", "swap_drag_scroll_notches")' in MAIN
    assert '"swap_drag_scroll_notches": self.swap_drag_scroll_notches' in MAIN


def test_interaction_page_has_a_toggle_for_drag_navigation_notches():
    layout = _method(MAIN, '    def _settings_layout(self, page=None):',
                     '    def _settings_control_rects(self,')
    assert 'stack("swap_drag_scroll_toggle")' in layout
    assert 'pressed("swap_drag_scroll_toggle")' in MAIN
    assert 'DRAG NAV' in MAIN and 'BACK/FORWARD SWAPPED' in MAIN


def test_drag_scroll_navigation_direction_and_arrows_both_swap():
    assert 'if self.swap_drag_scroll_notches:\n                                    direction = -direction' in MAIN
    assert MAIN.count('direction = sign * (-1.0 if self.swap_drag_scroll_notches else 1.0)') == 2


def test_hand_inference_and_scene_loop_use_lower_cpu_cadences():
    assert 'frame_interval = 1.0 / 18.0' in MAIN
    assert 'idle_frame_interval = 1.0 / 8.0' in MAIN
    assert 'paused_frame_interval = 1.0 / 2.0' in MAIN
    assert 'self.timestamp_ms = max(self.timestamp_ms + 1, int(now * 1000.0))' in MAIN
    assert 'time.sleep(min(interval - elapsed, 0.02))' in MAIN
    assert 'self.timer.start(50)' in MAIN
    assert 'desired_ms = 50 if hand_present or not self._scene_is_idle(now) else 100' in MAIN
    assert 'IDLE_REPAINT_INTERVAL = 0.33' in MAIN


def test_3d_render_is_downsampled_cached_and_does_not_copy_a_shadow_image_each_frame():
    method = _method(MAIN, '    def _paint_hand_mesh_3d(', '    def _report_hand_mesh_problem(')
    assert 'render_scale = 0.72' in method
    assert 'render_interval = 1.0 / 15.0' in method
    assert 'self._hand_mesh_frame_cache' in method
    assert 'QImage(image)' not in method
    assert 'CompositionMode_SourceIn' not in method


def test_linux_z_order_reassertion_stays_responsive_for_new_windows():
    method = _method(MAIN, '    def _maintain_native_topmost(self, now=None):',
                     '    def _sync_linux_fullscreen_geometry(self):')
    assert 'now-self._last_native_topmost_reassert <= 0.06' in method
    assert 'XRaiseWindow' in method
