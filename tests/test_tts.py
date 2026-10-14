#!/usr/bin/env python3
"""Play a short offline StreamTTS test phrase."""
from voice_services import KokoroTTS
import time

engine = KokoroTTS(status_callback=print)
if not engine.speak("Hello. This is a test of local, offline streaming text to speech."):
    raise SystemExit("Could not queue the TTS test.")
try:
    # Model setup and synthesis happen asynchronously; wait for queued speech.
    engine._pending.join()
    while engine.status.startswith(("Downloading", "Speaking")):
        time.sleep(0.1)
    if "unavailable" in engine.status.lower():
        raise SystemExit(engine.status)
finally:
    engine.close()
