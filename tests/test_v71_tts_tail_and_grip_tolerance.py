"""Regression checks for TTS utterance endings and more forgiving fist recognition."""
import unittest
from pathlib import Path

from voice_services import _complete_tts_utterance

ROOT = Path(__file__).resolve().parents[1]


class TtsEndpointTests(unittest.TestCase):
    def test_adds_terminal_punctuation_when_missing(self):
        self.assertEqual(_complete_tts_utterance("Voice input is off"), "Voice input is off.")

    def test_preserves_existing_sentence_ending(self):
        self.assertEqual(_complete_tts_utterance("Voice input is off!"), "Voice input is off!")
        self.assertEqual(_complete_tts_utterance("Already finished…"), "Already finished…")

    def test_keeps_closing_quotes_after_terminal_punctuation(self):
        self.assertEqual(_complete_tts_utterance('She said “hello”'), 'She said “hello.”')

    def test_stream_adds_tail_and_waits_for_device_latency(self):
        source = (ROOT / "streamtts" / "python" / "streamtts_python.py").read_text(encoding="utf-8")
        self.assertIn("self._text_queue.put(_UTTERANCE_END)", source)
        self.assertIn("def _enqueue_utterance_tail(self)", source)
        self.assertIn("drain_seconds = max(0.0, float(latency)) + 0.025", source)


class FistToleranceTests(unittest.TestCase):
    def test_recognizer_has_relaxed_curl_thresholds_and_thumb_margin(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("index_limit = 0.87 if edge_tolerant else 0.83", source)
        self.assertIn("other_limit = 0.90 if edge_tolerant else 0.86", source)
        self.assertIn("thumb_sticking_out(landmarks, threshold=-0.55)", source)


if __name__ == "__main__":
    unittest.main()
