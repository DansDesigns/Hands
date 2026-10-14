from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hand_mesh_3d as hm


def test_palm_frame_stays_orthonormal_as_across_axis_turns_edge_on():
    wrist = np.array([0.0, 0.0, 0.0])
    middle = np.array([0.0, 1.0, 0.0])
    previous = None
    frames = []
    for angle in np.linspace(0.0, np.pi, 41):
        across = np.array([np.cos(angle), 0.0, np.sin(angle)])
        index = middle + across * 0.5
        pinky = middle - across * 0.5
        frame = hm._palm_frame(wrist, index, middle, pinky, previous=previous)
        assert np.all(np.isfinite(frame))
        assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-6)
        assert np.linalg.det(frame) > 0.999
        if previous is not None:
            # No one-frame 180-degree roll of the thumb/palm basis.
            assert np.dot(previous[:, 1], frame[:, 1]) > 0.0
        frames.append(frame)
        previous = frame


def test_rendering_is_capped_for_lower_cpu_use():
    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    mesh_source = (ROOT / "hand_mesh_3d.py").read_text(encoding="utf-8")
    assert "frame_interval = 1.0 / 24.0" in main_source
    assert "self.timer.start(40)" in main_source
    assert "samples = 2 if max_samples >= 2 else 0" in mesh_source
