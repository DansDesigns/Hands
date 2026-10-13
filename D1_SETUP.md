# Liquid AI d1 local decision layer

The Windows installer downloads the default quantized **d1-omni-600M Q8** model and its matching multimodal projector (about **670 MB combined**) plus a compatible Windows CPU llama.cpp server. Spatial Desktop launches the local server automatically when needed; no separate terminal or manually running server is required.

## Settings

Open Glass Menu → Settings → **ENGINE** for local d1 and llama.cpp controls. The model row is pinned to d1-omni-600M Q8. Clicking it ensures that the default model is selected and requests a download if files are missing. The larger Q4 model has been removed from the options for now because its download was not completing reliably.

## Manual retry

From the project folder, run:

```bat
.venv\Scripts\python.exe install_d1_model.py 600m
```

The application remains usable without d1 if the download/runtime setup fails; existing voice command matching and local search remain the fallback. The server binds only to `127.0.0.1`.
