import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import hand_mesh_3d as hm

ROOT = Path(__file__).resolve().parents[1]


class JointNameTests(unittest.TestCase):
    def test_common_rig_names_map_to_mediapipe_joints(self):
        cases = {
            "wrist": "wrist",
            "index-finger-phalanx-proximal": "index-1",
            "middle-finger-phalanx-intermediate": "middle-2",
            "ring-finger-phalanx-distal": "ring-3",
            "pinky-finger-tip": "pinky-4",
            "thumb-metacarpal": "thumb-1",
            "thumb-phalanx-proximal": "thumb-2",
            "thumb-phalanx-distal": "thumb-3",
            "index-finger-metacarpal": "index-meta",
            "mixamorig:RightHand": "wrist",
            "mixamorig:LeftHandIndex2": "index-2",
            "mixamorig:RightHandThumb1": "thumb-1",
            "f_middle.03.R": "middle-3",
            "thumb.01.L": "thumb-1",
        }
        for name, expected in cases.items():
            self.assertEqual(hm.canonical_joint_name(name), expected, name)
        self.assertIsNone(hm.canonical_joint_name("Bone.007"))


class BundledModelTests(unittest.TestCase):
    def test_bundled_models_load_with_every_landmark_joint(self):
        for side in ("left", "right"):
            model = hm.HandModel(hm.BUNDLED_HANDS_DIR / f"generic-hand-{side}.glb")
            self.assertEqual(set(model.landmark_head), set(range(21)))
            self.assertGreater(model.palm_length, 0.0)
            self.assertTrue(0.0 <= float(model.fade.min()) < 0.05 and float(model.fade.max()) == 1.0)
        self.assertTrue((hm.BUNDLED_HANDS_DIR / "LICENSE-webxr-input-profiles.md").is_file())

    @staticmethod
    def _rest_targets(model):
        """Rest-pose landmarks as MediaPipe reports them: tips at the end of each finger."""
        target = np.array([model.landmark_head[i] for i in range(21)])
        for tip, dip in ((4, 3), (8, 7), (12, 11), (16, 15), (20, 19)):
            axis = target[tip]-target[dip]
            target[tip] = target[tip] + axis/np.linalg.norm(axis)*model.tip_cap[tip]
        return target

    def test_rest_pose_targets_reproduce_the_rest_mesh(self):
        model = hm.HandModel(hm.BUNDLED_HANDS_DIR / "generic-hand-left.glb")
        mats = hm.pose_joint_matrices(model, self._rest_targets(model))
        posed = model._skin(mats)
        rest = model._skin(model.rest_joint_mats) @ model.to_screen.T
        self.assertLess(float(np.abs(posed-rest).max()), 1e-4)

    def test_knuckles_land_on_landmarks_and_fingertips_end_on_tip_landmarks(self):
        model = hm.HandModel(hm.BUNDLED_HANDS_DIR / "generic-hand-right.glb")
        rest = self._rest_targets(model)
        # Scale, rotate and move the whole hand, then bend and lengthen the index tip.
        angle = 0.7
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        target = (rest-rest[0]) @ rot.T * 900.0 + np.array([800.0, 500.0, 0.0])
        target[8] = target[7] + 1.3*(target[8]-target[7]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
        mats = hm.pose_joint_matrices(model, target)
        for k, canonical in enumerate(model.canonical):
            landmark = hm.CANONICAL_LANDMARK.get(canonical)
            if landmark is None or landmark in (4, 8, 12, 16, 20):
                continue
            head_world = np.linalg.inv(model.to_screen) @ model.heads[k]
            delta = mats[k] @ np.linalg.inv(model.rest_joint_mats[k])
            moved = (delta @ np.r_[head_world, 1.0])[:3]
            self.assertLess(float(np.linalg.norm(moved-target[landmark])), 1e-6, canonical)
        # The far end of each finger's mesh sits on its tip landmark (TLI alignment).
        posed = model._skin(mats)
        dominant = model.joints[np.arange(len(model.joints)), np.argmax(model.weights, axis=1)]
        for tip, dip in ((4, 3), (8, 7)):
            own = [k for k, c in enumerate(model.canonical) if c and hm.CANONICAL_LANDMARK.get(c) in (tip, dip)]
            axis = (target[tip]-target[dip])/np.linalg.norm(target[tip]-target[dip])
            end = float(((posed[np.isin(dominant, own)]-target[tip]) @ axis).max())
            self.assertLess(abs(end), 0.5, f"tip {tip}")


class RenderStyleSettingsTests(unittest.TestCase):
    def test_render_style_round_trips_and_old_files_still_load(self):
        import main
        self.assertEqual(main.HAND_RENDER_STYLES, ("skeleton", "flat", "3d"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gestures.txt"
            with patch.object(main, "GESTURES_FILE", path):
                main.save_gesture_settings({"hand_render_style": "3d", "hand_skin_effect": True})
                self.assertIn("hand_render_style=3d", path.read_text(encoding="utf-8"))
                self.assertEqual(main.load_gesture_settings().get("hand_render_style"), "3d")
                # A settings file from before the three-way switch has only hand_skin_effect.
                path.write_text("hand_skin_effect=false\n", encoding="utf-8")
                settings = main.load_gesture_settings()
                self.assertNotIn("hand_render_style", settings)
                self.assertFalse(settings["hand_skin_effect"])

    def test_switch_click_and_paint_share_segment_geometry(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("self._render_style_segments("), 2)
        self.assertIn('stack("render_style")', source)


if __name__ == "__main__":
    unittest.main()
