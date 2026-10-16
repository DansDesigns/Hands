# Linux camera detection and backend selection

Spatial Desktop detects Intel IPU3/OV5670 hardware at runtime. On those systems,
`camera_backend.py` attempts a GStreamer `libcamerasrc` → `videoconvert` → BGR
`appsink` capture path. Other Linux systems retain the regular OpenCV/V4L2
capture path, and Windows retains DirectShow. The application capture object
supports the `read()`, `set()`, `isOpened()`, and `release()` methods used by
the tracker.

The Linux installer detects the available hardware and installs the matching
camera packages. It intentionally does not start a camera stream during install:
IPU3 stream experiments on some systems can hang the graphical session.

## Important limitation

Hardware detection and a working `cam -l` listing do not guarantee that frames
can be streamed. If `libcamerasrc` reports `Could not open any dma-buf provider`,
`not-negotiated`, or times out, that indicates a system libcamera/kernel/DMA
buffer integration problem. This backend reports the error and falls back to
OpenCV/V4L2; it does not silently claim the camera is working or attempt risky
privileged changes. A fix for that system-level DMA-buffer issue may still be
required on particular IPU3 kernels.

Do not run multiple camera streams concurrently. Close other camera applications
before launching Spatial Desktop.
