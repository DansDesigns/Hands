"""Regression tests for noisy 3D finger joints and small hand scales."""
import unittest
from pathlib import Path

import numpy as np

import hand_mesh_3d as hm

ROOT = Path(__file__).resolve().parents[1]


class BoneAlignmentTests(unittest.TestCase):
    def test_collapsed_target_segment_does_not_create_arbitrary_half_turn(self):
        rotation = hm._align(np.array([0.0, 1.0, 0.0]), np.zeros(3))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_true_half_turn_uses_palm_relative_axis(self):
        rotation = hm._align(
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            preferred_axes=(np.array([0.0, 0.0, 1.0]),),
        )
        np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [-1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)

    def test_palm_frame_stays_orthonormal_near_edge_on_across_axis(self):
        wrist = np.array([0.0, 0.0, 0.0])
        index = np.array([0.001, 0.8, 0.0])
        middle = np.array([0.0, 1.0, 0.0])
        pinky = np.array([-0.001, 0.8, 0.0])
        first = hm._palm_frame(wrist, index, middle, pinky)
        # Perturb the nearly collapsed across-axis as tracking commonly does.
        second = hm._palm_frame(
            wrist, index + np.array([0.0, 0.0, 0.0002]), middle,
            pinky - np.array([0.0, 0.0, 0.0002]), previous=first,
        )
        np.testing.assert_allclose(second.T @ second, np.eye(3), atol=1e-6)
        self.assertGreater(float(np.dot(first[:, 1], second[:, 1])), 0.0)


class SmallScalePoseTests(unittest.TestCase):
    @staticmethod
    def _rest_targets(model):
        target = np.array([model.landmark_head[i] for i in range(21)], dtype=np.float64)
        for tip, dip in ((4, 3), (8, 7), (12, 11), (16, 15), (20, 19)):
            axis = target[tip] - target[dip]
            target[tip] += axis / np.linalg.norm(axis) * model.tip_cap[tip]
        return target

    def test_finger_shape_scales_linearly_without_changing_proportions(self):
        model = hm.HandModel(hm.BUNDLED_HANDS_DIR / "generic-hand-right.glb")
        target = self._rest_targets(model)
        # Make the index and thumb pose more demanding than the neutral model pose.
        target[8] = target[7] + 1.3 * (target[8] - target[7])
        target[4] = target[3] + 1.2 * (target[4] - target[3])
        normal = model._skin(hm.pose_joint_matrices(model, target))
        scale = 0.1
        small_target = target[0] + scale * (target - target[0])
        small = model._skin(hm.pose_joint_matrices(model, small_target))
        expected = target[0] + scale * (normal - target[0])
        np.testing.assert_allclose(small, expected, atol=1e-5, rtol=1e-5)

    def test_collapsed_thumb_joint_keeps_pose_finite(self):
        model = hm.HandModel(hm.BUNDLED_HANDS_DIR / "generic-hand-left.glb")
        target = self._rest_targets(model)
        target[2] = target[1]
        mats = hm.pose_joint_matrices(model, target)
        self.assertTrue(np.isfinite(mats).all())
        self.assertTrue(np.isfinite(model._skin(mats)).all())

    def test_nearly_collapsed_thumb_direction_is_treated_as_tracking_noise(self):
        model = hm.HandModel(hm.BUNDLED_HANDS_DIR / "generic-hand-left.glb")
        target = self._rest_targets(model)
        palm = np.linalg.norm(target[9]-target[0])
        target[2] = target[1] + np.array([0.0, 0.0, palm*0.005])
        mats = hm.pose_joint_matrices(model, target)
        self.assertTrue(np.isfinite(mats).all())
        self.assertTrue(np.isfinite(model._skin(mats)).all())
        source = (ROOT / "hand_mesh_3d.py").read_text(encoding="utf-8")
        self.assertIn("target_palm_length*0.025", source)

    def test_small_hand_scale_rendering_and_scale_range_available(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('("hand_scale", self.hand_visual_scale, 0.1, 1.0, "HAND SCALE")', source)
        # The software-rendering optimisation removed the duplicate shadow-image copy.
        self.assertIn('render_scale = 0.72', source)
        self.assertNotIn('CompositionMode_SourceIn', source)


if __name__ == "__main__":
    unittest.main()
