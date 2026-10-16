3D HAND MODEL FOLDERS
=====================

The Hand Settings page's "3D HAND MODEL" selector lists each subfolder here
that contains at least one .glb or .gltf file. Keep the two sides for a model
set together in one folder.

Accepted filenames:
  left.glb / right.glb
  hand-left.glb / hand-right.glb
  generic-hand-left.glb / generic-hand-right.glb
  model-left.glb / model-right.glb

The same filename patterns also work with .gltf. If the folder contains only
one side, Spatial Desktop mirrors that model for the missing side.

Example:
  assets/hands/
    generic-hand-left.glb
    generic-hand-right.glb
    soft-hands/
      left.glb
      right.glb
    single-hand/
      hand-right.glb

The "Default" option uses the generic-hand-left/right.glb files directly in
assets/hands/. Model changes are saved in ui_settings.json and apply the next
time the 3D renderer is initialized.
