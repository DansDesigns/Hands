#!/usr/bin/env python3
"""Play a short offline Kokoro test phrase if the local model is installed."""
from voice_services import KokoroTTS
import time

engine = KokoroTTS(status_callback=print)
print(engine.status)
if not engine.ready:
    raise SystemExit("TTS is not ready. Install requirements-voice.txt, run install_voice_models.py, and check eSpeak NG if loading fails.")
if not engine.speak("Hello. This is a test of local, offline text to speech."):
    raise SystemExit("Could not queue the TTS test.")
# The TTS worker plays asynchronously. Keep the process alive until it finishes.
while engine._thread is not None and engine._thread.is_alive():
    time.sleep(0.1)
engine.close()
