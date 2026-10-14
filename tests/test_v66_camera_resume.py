import threading
import time
import unittest

import numpy as np

import main


class FakeCapture:
    def __init__(self, grab_ok=True, frames=None, grab_delay=0.0):
        self.grab_ok = grab_ok
        self.frames = frames
        self.grab_delay = grab_delay
        self.released = False
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        if self.grab_delay and self.grabs >= 3:
            delay, self.grab_delay = self.grab_delay, 0.0   # one long stall, like a suspend
            time.sleep(delay)
        time.sleep(0.005)
        return self.grab_ok

    def retrieve(self):
        if self.frames is None:
            return True, np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        return True, self.frames.copy()

    def release(self):
        self.released = True


def bare_tracker(cap):
    tracker = object.__new__(main.Tracker)
    tracker.cap = cap
    tracker.camera_index = 0
    tracker._reopen_failures = 0
    tracker.running = True
    tracker.hand_tracking_enabled = False      # no MediaPipe needed for these tests
    tracker.hands_data = []
    tracker._candidate_streak = {"left": 0, "right": 0}
    tracker._candidate_center = {"left": None, "right": None}
    tracker._candidate_misses = {"left": 0, "right": 0}
    tracker.frame_id = 0
    tracker.timestamp_ms = 0
    tracker.latest_frame = None
    tracker.lock = threading.Lock()
    return tracker


class CameraRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # OpenCV's first resize/flip can take a second to initialise; keep that out of
        # the timing-based tests below.
        main.cv2.resize(main.cv2.flip(np.zeros((360, 640, 3), np.uint8), 1), (480, 270))

    def run_loop_until_reopen(self, tracker, timeout=4.0, gap=2.0):
        reasons = []

        def fake_reopen(reason):
            reasons.append(reason)
            tracker.running = False
            return True

        tracker._reopen_camera = fake_reopen
        tracker.CAMERA_STALL_SECONDS = 0.15
        tracker.CAMERA_RESUME_GAP_SECONDS = gap
        thread = threading.Thread(target=tracker._loop, daemon=True)
        thread.start()
        thread.join(timeout)
        tracker.running = False
        return reasons

    def test_camera_that_stops_delivering_is_reopened(self):
        reasons = self.run_loop_until_reopen(bare_tracker(FakeCapture(grab_ok=False)))
        self.assertEqual(len(reasons), 1)
        self.assertIn("stopped delivering", reasons[0])

    def test_frozen_picture_is_reopened(self):
        frame = np.full((360, 640, 3), 77, dtype=np.uint8)
        reasons = self.run_loop_until_reopen(bare_tracker(FakeCapture(frames=frame)))
        self.assertEqual(len(reasons), 1)
        self.assertIn("frozen", reasons[0])

    def test_long_pause_like_sleep_resume_reopens_at_once(self):
        reasons = self.run_loop_until_reopen(bare_tracker(FakeCapture(grab_delay=0.8)), gap=0.5)
        self.assertEqual(len(reasons), 1)
        self.assertIn("sleep/resume", reasons[0])

    def test_live_camera_is_left_alone(self):
        tracker = bare_tracker(FakeCapture())
        reasons = []
        tracker._reopen_camera = lambda reason: reasons.append(reason) or True
        tracker.CAMERA_STALL_SECONDS = 0.15
        tracker.CAMERA_RESUME_GAP_SECONDS = 2.0
        thread = threading.Thread(target=tracker._loop, daemon=True)
        thread.start()
        time.sleep(0.8)
        tracker.running = False
        thread.join(1.0)
        self.assertEqual(reasons, [])
        self.assertGreater(tracker.frame_id, 0)

    def test_reopen_releases_old_handle_and_adopts_a_delivering_camera(self):
        old, new = FakeCapture(), FakeCapture()
        tracker = bare_tracker(old)
        opened = []
        tracker._open_camera = lambda index: opened.append(index) or new
        self.assertTrue(tracker._reopen_camera("test"))
        self.assertTrue(old.released)
        self.assertIs(tracker.cap, new)
        self.assertEqual(opened, [0])

    def test_reopen_tries_other_indexes_only_after_repeated_failures(self):
        tracker = bare_tracker(FakeCapture())
        opened = []
        tracker._open_camera = lambda index: opened.append(index) or None
        self.assertFalse(tracker._reopen_camera("test"))
        self.assertFalse(tracker._reopen_camera("test"))
        self.assertEqual(opened, [0, 0])
        self.assertIsNone(tracker.cap)
        new = FakeCapture()
        tracker._open_camera = lambda index: opened.append(index) or (new if index == 2 else None)
        self.assertTrue(tracker._reopen_camera("test"))
        self.assertIs(tracker.cap, new)
        self.assertEqual(opened[2:], [0, 1, 2])
        self.assertEqual(tracker.camera_index, 0)   # the preferred camera stays first


if __name__ == "__main__":
    unittest.main()
