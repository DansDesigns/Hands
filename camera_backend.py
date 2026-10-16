"""Hardware-aware camera capture for Spatial Desktop.

Conventional webcams keep using OpenCV/V4L2. Intel IPU3 cameras are exposed
through libcamera, so use GStreamer's libcamerasrc/appsink without requiring a
custom OpenCV build. This module does not run camera tests during installation.
"""
from __future__ import annotations

import glob
import os
import sys


def _is_ipu3_system() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/modules", "r", encoding="utf-8", errors="ignore") as f:
            modules = f.read().lower()
        if any(x in modules for x in ("ipu3_cio2", "ipu3_imgu", "ov5670")):
            return True
    except OSError:
        pass
    return (os.path.exists("/sys/bus/platform/drivers/ipu3-cio2") or
            os.path.exists("/sys/bus/pci/drivers/ipu3-cio2") or
            os.path.exists("/sys/bus/i2c/devices/i2c-INT3479:01"))


def _enable_system_gi_paths() -> None:
    """Make distro PyGObject visible from a venv where system-site-packages is off."""
    for path in ("/usr/lib/python3/dist-packages", *glob.glob("/usr/lib/python3*/dist-packages"),
                 *glob.glob("/usr/lib/*/python3/dist-packages")):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


class LibcameraGstCapture:
    """Small cv2.VideoCapture-compatible wrapper around libcamerasrc/appsink."""
    def __init__(self, width=640, height=360, fps=30):
        _enable_system_gi_paths()
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        self.Gst = Gst
        # Do not constrain the source caps here. Some IPU3/libcamera versions
        # advertise incomplete frame-duration ranges for the OV5670 sensor; asking
        # for a synthetic 640x360@30 mode can trigger a libcamera std::clamp
        # assertion (which aborts the whole Python process before exceptions run).
        # Let libcamera choose a sensor-supported mode, then convert downstream.
        self.pipeline = Gst.parse_launch(
            "libcamerasrc ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=spatial_sink max-buffers=1 drop=true sync=false"
        )
        self.sink = self.pipeline.get_by_name("spatial_sink")
        print("[camera] Starting libcamera with sensor-native mode selection (no forced width/FPS caps).", flush=True)
        self.pipeline.set_state(Gst.State.PLAYING)
        state, _, _ = self.pipeline.get_state(3 * Gst.SECOND)
        if state == Gst.StateChangeReturn.FAILURE:
            self.release()
            raise RuntimeError("GStreamer could not start the Intel IPU3/libcamera camera pipeline.")
        self._opened = True

    def isOpened(self):
        return self._opened

    def set(self, *_args):
        # Resolution and frame rate are set in the GStreamer caps.
        return True

    def read(self):
        sample = self.sink.emit("try-pull-sample", 500 * self.Gst.MSECOND)
        if sample is None:
            return False, None
        import numpy as np
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            frame = np.frombuffer(mapped.data, dtype=np.uint8)
            # GStreamer BGR caps produce tightly packed BGR rows on this path.
            frame = frame.reshape((height, width, 3)).copy()
            return True, frame
        except (ValueError, TypeError):
            return False, None
        finally:
            buffer.unmap(mapped)

    def release(self):
        self._opened = False
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.set_state(self.Gst.State.NULL)


def open_camera(cv2, width=640, height=360):
    """Return (capture, backend_name), preferring the hardware-native backend."""
    errors = []
    if _is_ipu3_system():
        try:
            capture = LibcameraGstCapture(width, height)
            print("[camera] Using Intel IPU3 through libcamera/GStreamer appsink.", flush=True)
            return capture, "libcamera-gstreamer"
        except Exception as exc:
            errors.append(f"libcamera/GStreamer: {exc}")
            print("[camera] IPU3/libcamera backend failed: " + str(exc), flush=True)
            print("[camera] Trying ordinary OpenCV/V4L2 as a fallback.", flush=True)

    indices = [0]
    # IPU3 /dev/video nodes are often ISP subdevices, not directly usable
    # webcam nodes. Only probe index zero as a fallback; avoid opening every
    # node because this can lock the camera or destabilise the compositor.
    for index in indices:
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            print(f"[camera] Using OpenCV camera index {index}.", flush=True)
            return capture, f"opencv-index-{index}"
        capture.release()

    detail = "; ".join(errors)
    message = "Could not open a usable camera."
    if _is_ipu3_system():
        message += (" Intel IPU3 hardware was detected, but this camera needs a working "
                    "libcamera/GStreamer stream. Check `cam -l` and the system journal; "
                    "do not repeatedly launch multiple camera tests because IPU3 can hang the desktop.")
    if detail:
        message += " Backend detail: " + detail
    raise RuntimeError(message)
