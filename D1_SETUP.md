# Liquid AI d1 local decision layer

The Windows and Linux installers download the default quantized **d1-omni-600M Q8** model and its matching multimodal projector (about **670 MB combined**) plus a llama.cpp server matched to the machine: the official CPU build for Windows x64/ARM64 or Linux x64/ARM64/s390x. On any other architecture, or if a prebuilt server cannot run (for example an older or musl-based Linux), the installer compiles `llama-server` locally when `cmake` and a C++ compiler are installed. Spatial Desktop launches the local server automatically when needed; no separate terminal or manually running server is required.

## Settings

Open Glass Menu → Settings → **ENGINE** for local d1 and llama.cpp controls. The model row is pinned to d1-omni-600M Q8. Clicking it ensures that the default model is selected and requests a download if files are missing. The larger Q4 model has been removed from the options for now because its download was not completing reliably.

## Manual retry

From the project folder, run:

```bat
.venv\Scripts\python.exe install_d1_model.py 600m
```

On Linux:

```sh
.venv/bin/python install_d1_model.py 600m
```

Add `--force` to replace an existing server. To reuse a `llama-server` you already have, set `SPATIAL_DESKTOP_LLAMA_SERVER` to its path before running the installer.

The application remains usable without d1 if the download/runtime setup fails; existing voice command matching and local search remain the fallback. The server binds only to `127.0.0.1`.
