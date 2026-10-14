from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hand_mapping_uses_physical_screen_not_usable_widget_area():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    start = source.index("    def interactive_visual_point(self, x, y):")
    end = source.index("    def interactive_hand_anchor", start)
    method = source[start:end]
    assert "QApplication.primaryScreen()" in method
    assert "screen.geometry()" in method
    assert "screen_width-1" in method
    assert "screen_height-1" in method
    assert "self.width()-1" not in method
    assert "self.height()-1" not in method
