"""Regression coverage for the two-hand thumb-to-middle pinch zoom gesture."""
from pathlib import Path

from gesture_rules import (
    thumb_middle_pinch_pose,
    opposite_diagonal_corner_assignment,
    resize_rect_from_corner_deltas,
    resize_rect_from_distance_scale,
    classify_middle_pinch_window_action,
)


def test_middle_pinch_requires_middle_tip_to_be_closer_than_index():
    assert thumb_middle_pinch_pose(
        normalized_middle_gap=0.22,
        normalized_index_gap=0.55,
        finger_straightness=[0.82, 0.47, 0.78, 0.74],
    )
    assert not thumb_middle_pinch_pose(
        normalized_middle_gap=0.22,
        normalized_index_gap=0.20,
        finger_straightness=[0.82, 0.47, 0.78, 0.74],
    )


def test_middle_pinch_rejects_wide_gap_and_closed_fist():
    extended = [0.82, 0.47, 0.78, 0.74]
    assert not thumb_middle_pinch_pose(
        normalized_middle_gap=0.71, normalized_index_gap=0.8,
        finger_straightness=extended,
    )
    assert not thumb_middle_pinch_pose(
        normalized_middle_gap=0.20, normalized_index_gap=0.8,
        finger_straightness=[0.45, 0.40, 0.50, 0.48],
    )


def test_edge_tolerance_only_modestly_relaxes_middle_pinch_gap():
    values = dict(normalized_middle_gap=0.54, normalized_index_gap=0.8,
                  finger_straightness=[0.8, 0.5, 0.75, 0.72])
    assert not thumb_middle_pinch_pose(**values)
    assert thumb_middle_pinch_pose(**values, edge_tolerant=True)


def test_opposite_window_corners_recognised_in_both_diagonals():
    rect = (100, 100, 900, 700)
    assert opposite_diagonal_corner_assignment(rect, (105, 106), (894, 694)) == {
        "left": "tl", "right": "br"
    }
    assert opposite_diagonal_corner_assignment(rect, (895, 105), (105, 695)) == {
        "left": "tr", "right": "bl"
    }


def test_same_side_or_non_corner_points_do_not_enter_resize_mode():
    rect = (100, 100, 900, 700)
    assert opposite_diagonal_corner_assignment(rect, (104, 105), (105, 110)) is None
    assert opposite_diagonal_corner_assignment(rect, (300, 100), (900, 400)) is None



def test_corner_zone_classifier_uses_inset_opposite_diagonal_quadrants():
    rect = (100, 100, 900, 700)
    # 25% inward from opposite corners still means application-window resizing.
    assert classify_middle_pinch_window_action(
        rect, (300, 250), (700, 550)) == ("resize", {"left": "tl", "right": "br"})
    assert classify_middle_pinch_window_action(
        rect, (700, 250), (300, 550)) == ("resize", {"left": "tr", "right": "bl"})


def test_non_corner_positions_inside_same_window_select_content_zoom():
    rect = (100, 100, 900, 700)
    assert classify_middle_pinch_window_action(
        rect, (320, 300), (680, 500)) == ("zoom", None)
    assert classify_middle_pinch_window_action(
        rect, (440, 250), (560, 550)) == ("zoom", None)


def test_pinches_outside_same_window_do_not_accidentally_select_corner_resize():
    rect = (100, 100, 900, 700)
    assert classify_middle_pinch_window_action(
        rect, (80, 100), (700, 550)) == ("zoom", None)

def test_two_pinned_corners_expand_and_contract_window():
    rect = (100, 100, 900, 700)
    start = {"left": (100, 100), "right": (900, 700)}
    corners = {"left": "tl", "right": "br"}
    expanded = resize_rect_from_corner_deltas(
        rect, start, {"left": (50, 40), "right": (1000, 800)}, corners)
    assert expanded == (50.0, 40.0, 1000.0, 800.0)
    contracted = resize_rect_from_corner_deltas(
        rect, start, {"left": (160, 145), "right": (820, 635)}, corners)
    assert contracted == (160.0, 145.0, 820.0, 635.0)


def test_resize_keeps_minimum_window_dimensions():
    rect = (100, 100, 900, 700)
    start = {"left": (100, 100), "right": (900, 700)}
    result = resize_rect_from_corner_deltas(
        rect, start, {"left": (880, 680), "right": (120, 120)},
        {"left": "tl", "right": "br"}, min_width=260, min_height=160)
    assert result is not None
    assert result[2] - result[0] >= 260
    assert result[3] - result[1] >= 160


def test_resize_supports_other_diagonal_corner_pair():
    rect = (100, 100, 900, 700)
    start = {"left": (900, 100), "right": (100, 700)}
    result = resize_rect_from_corner_deltas(
        rect, start, {"left": (940, 60), "right": (60, 760)},
        {"left": "tr", "right": "bl"})
    assert result == (60.0, 60.0, 940.0, 760.0)


def test_default_window_resize_scales_around_center_as_hands_separate():
    rect = (100, 100, 900, 700)
    expanded = resize_rect_from_distance_scale(rect, 400, 600)
    assert expanded == ( -100.0, -50.0, 1100.0, 850.0)
    contracted = resize_rect_from_distance_scale(rect, 400, 200)
    assert contracted == (300.0, 250.0, 700.0, 550.0)


def test_default_window_resize_scales_are_bounded_and_validate_inputs():
    rect = (0, 0, 800, 600)
    expanded = resize_rect_from_distance_scale(rect, 100, 10_000, max_scale=2.0)
    assert expanded == (-400.0, -300.0, 1200.0, 900.0)
    assert resize_rect_from_distance_scale(rect, 0, 100) is None
    assert resize_rect_from_distance_scale((0, 0, 0, 5), 10, 20) is None


def test_main_wires_the_shared_buffer_timer_and_both_visual_layers():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert 'now-float(state["started_at"]) >= self.pinch_buffer_seconds' in source
    assert 'self._process_middle_pinch_zoom(cursor_points, poses, now)' in source
    assert 'self._paint_middle_pinch_zoom_timer(painter, now)' in source
    assert 'owner._paint_middle_pinch_zoom_timer(painter, now)' in source


def test_main_wires_content_zoom_and_native_window_resize_paths():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert 'pyautogui.keyDown(modifier, _pause=False)' in source
    assert 'pyautogui.scroll(1 if direction > 0 else -1, _pause=False)' in source
    assert 'pyautogui.moveTo(int(screen_point.x()), int(screen_point.y())' not in source
    assert 'self._resize_native_window(state.get("hwnd")' in source
    assert 'classify_middle_pinch_window_action(' in source
    assert 'resize_rect_from_distance_scale(' in source
    assert 'classify_middle_pinch_window_action(' in source
    assert 'corner_zone_ratio=0.25' in source


def test_content_mode_is_selected_by_window_position_not_app_process():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    rules = (Path(__file__).resolve().parents[1] / "gesture_rules.py").read_text(encoding="utf-8")
    assert 'self._native_window_prefers_content_zoom' not in source
    assert 'browser_processes' not in source
    assert 'classify_middle_pinch_window_action' in rules
    assert 'return "zoom", None' in rules


def test_middle_pinch_does_not_steal_fist_grab_or_unpaired_single_hand_frames():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _process_middle_pinch_zoom(self, cursor_points, poses, now):")
    end = source.index("    def _paint_middle_pinch_zoom_timer(self, painter, now=None):", start)
    method = source[start:end]
    # Grip/fist wins when its folded middle fingertip is close to the thumb, and the
    # built-in two-hand zoom may only consume frames once both pinches start together.
    assert 'and not poses.get(side, {}).get("fist", False)' in method
    assert 'and not both_middle:' in method
    assert 'return False' in method
