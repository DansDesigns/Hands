"""Real 3D hand rendering: skinned glTF hand meshes posed from MediaPipe landmarks.

The hand model is a rigged glTF/GLB mesh. Its joints are matched to MediaPipe's 21
landmarks by name (WebXR hand-joint names, plus common Blender/Mixamo-style names), each
bone is placed at its landmark and rotated to point at the next one, and the mesh is
skinned on the GPU. Rendering happens offscreen through moderngl (OpenGL); the result is
read back as a QImage and composited onto the transparent overlay by the caller.

Everything here is optional: if moderngl or an OpenGL context is unavailable, the caller
falls back to the 2D "flat" renderer.

Bundled models: assets/hands/generic-hand-{left,right}.glb from the W3C/Immersive Web
WebXR Input Profiles project (MIT licence, see assets/hands/LICENSE-webxr-input-profiles.md).
Custom models: put models/hands/custom-left.glb and/or custom-right.glb (git-ignored); if
only one side is provided it is mirrored for the other hand. Bone names are matched
automatically (WebXR, Blender and Mixamo-style names); otherwise add custom-right.joints.json
mapping bone names to wrist / <finger>-meta / <finger>-1..4.
"""
from __future__ import annotations

import json
import math
import re
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# Bundled models live in a tracked folder; user models go in models/ (git-ignored).
BUNDLED_HANDS_DIR = ROOT / "assets" / "hands"
CUSTOM_HANDS_DIR = ROOT / "models" / "hands"

# MediaPipe landmark for each canonical joint. Metacarpals have no landmark of their
# own; they ride rigidly with the palm.
CANONICAL_LANDMARK = {
    "wrist": 0,
    "thumb-1": 1, "thumb-2": 2, "thumb-3": 3, "thumb-4": 4,
    "index-1": 5, "index-2": 6, "index-3": 7, "index-4": 8,
    "middle-1": 9, "middle-2": 10, "middle-3": 11, "middle-4": 12,
    "ring-1": 13, "ring-2": 14, "ring-3": 15, "ring-4": 16,
    "pinky-1": 17, "pinky-2": 18, "pinky-3": 19, "pinky-4": 20,
}
# Landmark each landmark-driven bone points toward.
NEXT_LANDMARK = {1: 2, 2: 3, 3: 4, 5: 6, 6: 7, 7: 8, 9: 10, 10: 11, 11: 12,
                 13: 14, 14: 15, 15: 16, 17: 18, 18: 19, 19: 20}
PREVIOUS_LANDMARK = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}

_FINGER_WORDS = (
    ("thumb", ("thumb", "pollex")),
    ("index", ("index", "pointer")),
    ("middle", ("middle",)),
    ("ring", ("ring",)),
    ("pinky", ("pinky", "little", "small", "pinkie")),
)


def canonical_joint_name(name: str) -> str | None:
    """Map a rig's joint name to 'wrist', '<finger>-meta' or '<finger>-<1..4>'."""
    n = (name or "").lower()
    n = re.sub(r"(mixamorig[:_]?|[._ -](l|r)$|^(l|r)[._ -]|left|right)", " ", n)
    finger = None
    for canonical, words in _FINGER_WORDS:
        if any(word in n for word in words):
            finger = canonical
            break
    if finger is None:
        return "wrist" if ("wrist" in n or re.search(r"\bhand\b", n)) else None
    if "metacarpal" in n:
        # WebXR: thumb-metacarpal is the CMC (landmark 1); other metacarpals are palm bones.
        return f"{finger}-1" if finger == "thumb" else f"{finger}-meta"
    words = {"proximal": 1, "intermediate": 2, "distal": 3, "tip": 4, "end": 4}
    for word, ordinal in words.items():
        if word in n:
            if finger == "thumb" and word == "distal":
                return "thumb-3"
            if finger == "thumb" and word == "proximal":
                return "thumb-2"
            return f"{finger}-{ordinal}"
    digits = re.findall(r"(\d+)", n)
    if digits:
        ordinal = int(digits[-1])
        if 1 <= ordinal <= 4:
            return f"{finger}-{ordinal}"
    return None


# --------------------------------------------------------------------------------------
# glTF loading
# --------------------------------------------------------------------------------------

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_gltf(path: Path):
    data = path.read_bytes()
    if data[:4] == b"glTF":
        offset, chunks = 12, {}
        while offset + 8 <= len(data):
            length, kind = struct.unpack_from("<I4s", data, offset)
            chunks[kind] = data[offset+8:offset+8+length]
            offset += 8 + length
        gltf = json.loads(chunks[b"JSON"].decode("utf-8"))
        buffers = [chunks.get(b"BIN\x00", b"")]
        for extra in gltf.get("buffers", [])[1:]:
            buffers.append((path.parent / extra["uri"]).read_bytes())
        return gltf, buffers
    gltf = json.loads(data.decode("utf-8"))
    buffers = []
    for buffer in gltf.get("buffers", []):
        uri = buffer.get("uri", "")
        if uri.startswith("data:"):
            import base64
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            buffers.append((path.parent / uri).read_bytes())
    return gltf, buffers


def _accessor(gltf, buffers, index):
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT[accessor["componentType"]])
    width = _WIDTH[accessor["type"]]
    count = accessor["count"]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 0) or dtype.itemsize*width
    raw = buffers[view.get("buffer", 0)]
    if stride == dtype.itemsize*width:
        array = np.frombuffer(raw, dtype=dtype, count=count*width, offset=start).reshape(count, width)
    else:
        rows = [np.frombuffer(raw, dtype=dtype, count=width, offset=start+i*stride) for i in range(count)]
        array = np.stack(rows)
    array = array.astype(np.float32 if dtype == np.float32 else np.int64)
    if accessor.get("normalized") and dtype != np.float32:
        array = array.astype(np.float32) / float(np.iinfo(dtype).max)
    return array


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _local_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(node.get("rotation", [0, 0, 0, 1])) @ np.diag(node.get("scale", [1, 1, 1]))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


class HandModel:
    """A skinned hand mesh in its rest pose, converted to screen-style axes."""

    # glTF is y-up with +z toward the viewer; the overlay uses y-down with +z away from
    # the viewer. diag(1, -1, -1) is a proper rotation, so handedness is preserved.
    TO_SCREEN = np.diag([1.0, -1.0, -1.0])

    def __init__(self, path: Path, mirror: bool = False):
        gltf, buffers = _read_gltf(path)
        nodes = gltf["nodes"]
        mesh_node = next(i for i, n in enumerate(nodes) if "mesh" in n and "skin" in n)
        skin = gltf["skins"][nodes[mesh_node]["skin"]]
        primitive = gltf["meshes"][nodes[mesh_node]["mesh"]]["primitives"][0]
        attributes = primitive["attributes"]
        positions = _accessor(gltf, buffers, attributes["POSITION"]).astype(np.float64)
        normals = (_accessor(gltf, buffers, attributes["NORMAL"]).astype(np.float64)
                   if "NORMAL" in attributes else None)
        joints = _accessor(gltf, buffers, attributes["JOINTS_0"]).astype(np.int64)
        weights = _accessor(gltf, buffers, attributes["WEIGHTS_0"]).astype(np.float64)
        weights /= np.maximum(1e-9, weights.sum(axis=1, keepdims=True))
        if "indices" in primitive:
            indices = _accessor(gltf, buffers, primitive["indices"]).reshape(-1).astype(np.uint32)
        else:
            indices = np.arange(len(positions), dtype=np.uint32)

        # Global rest transforms of every node.
        parent = {}
        for i, node in enumerate(nodes):
            for child in node.get("children", []):
                parent[child] = i
        globals_ = {}

        def global_matrix(i):
            if i not in globals_:
                local = _local_matrix(nodes[i])
                globals_[i] = global_matrix(parent[i]) @ local if i in parent else local
            return globals_[i]

        joint_nodes = skin["joints"]
        if "inverseBindMatrices" in skin:
            ibm = _accessor(gltf, buffers, skin["inverseBindMatrices"]).astype(np.float64)
            ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            ibm = np.stack([np.eye(4)]*len(joint_nodes))
        # Joint matrices for the rest pose: vertices skinned by these give the rest
        # mesh in world space, whatever bind pose the file was exported with.
        self.rest_joint_mats = np.stack([global_matrix(j) @ ibm[k] for k, j in enumerate(joint_nodes)])
        heads = np.array([global_matrix(j)[:3, 3] for j in joint_nodes])

        flip = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        self.to_screen = self.TO_SCREEN @ flip
        self.positions = positions.astype(np.float32)
        if normals is None:
            normals = self._vertex_normals(positions, indices)
        self.normals = normals.astype(np.float32)
        self.joints = joints.astype(np.int32)
        self.weights = weights.astype(np.float32)
        # Mirroring flips triangle winding; rendering is double-sided so it is harmless.
        self.indices = indices
        self.joint_count = len(joint_nodes)
        self.heads = heads @ self.to_screen.T          # rest joint heads, screen axes
        self.names = [nodes[j].get("name", f"joint{k}") for k, j in enumerate(joint_nodes)]
        self.canonical = [canonical_joint_name(name) for name in self.names]
        # Optional <model>.joints.json: {"BoneName": "index-1", ...} for rigs whose bone
        # names aren't recognised. Values: wrist, <finger>-meta or <finger>-1..4, where
        # finger is thumb/index/middle/ring/pinky (1 = knuckle bone, 4 = tip).
        mapping_path = path.with_suffix(".joints.json")
        if mapping_path.is_file():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            self.canonical = [mapping.get(name, canonical) for name, canonical in zip(self.names, self.canonical)]

        found = {c: k for k, c in enumerate(self.canonical) if c}
        missing = [c for c in ("wrist", "index-1", "middle-1", "pinky-1") if c not in found]
        if missing:
            raise ValueError(f"hand rig is missing joints: {', '.join(missing)}")
        self.joint_for = found
        self.landmark_head = {}
        for canonical, landmark in CANONICAL_LANDMARK.items():
            if canonical in found:
                self.landmark_head[landmark] = self.heads[found[canonical]]
        # Palm size and mesh "reach" beyond the joints, for scaling and bounding boxes.
        self.palm_length = float(np.linalg.norm(self.landmark_head[9]-self.landmark_head[0]))
        rest_world = self._skin(self.rest_joint_mats)
        rest_screen = rest_world @ self.to_screen.T
        distances = np.min(np.linalg.norm(rest_screen[:, None, :]-self.heads[None, :, :], axis=2), axis=1)
        self.reach = float(np.percentile(distances, 99.5))
        # How far the mesh extends past each fingertip joint, along the last bone.
        # Rigs differ (WebXR places the "tip" joint mid-pad), but MediaPipe's tip
        # landmark is the very end of the finger, so posing matches the mesh end to it.
        dominant = self.joints[np.arange(len(self.joints)), np.argmax(self.weights, axis=1)]
        self.tip_cap = {}
        for tip, dip in ((4, 3), (8, 7), (12, 11), (16, 15), (20, 19)):
            if tip not in self.landmark_head or dip not in self.landmark_head:
                continue
            axis = self.landmark_head[tip]-self.landmark_head[dip]
            length = float(np.linalg.norm(axis))
            own = [k for k, c in enumerate(self.canonical) if c and CANONICAL_LANDMARK.get(c) in (tip, dip)]
            verts = rest_screen[np.isin(dominant, own)]
            if length > 1e-9 and len(verts):
                beyond = float(((verts-self.landmark_head[tip]) @ (axis/length)).max())
                self.tip_cap[tip] = max(0.0, beyond)
        # Fade the forearm out just past the wrist so the mesh never ends in a hard,
        # open "sleeve" edge (1 = fully visible, 0 = transparent).
        up = self.landmark_head[9]-self.landmark_head[0]
        up = up/max(1e-9, float(np.linalg.norm(up)))
        along = (rest_screen-self.landmark_head[0]) @ up / max(1e-9, self.palm_length)
        # Measured from this mesh: transparent at the far end of its forearm, opaque
        # just below the wrist (meshes without a forearm are left fully visible).
        sleeve_end = float(along.min())
        if sleeve_end < -0.05:
            t = np.clip((along-sleeve_end)/max(1e-6, -0.02-sleeve_end), 0.0, 1.0)
        else:
            t = np.ones_like(along)
        self.fade = (t*t*(3.0-2.0*t)).astype(np.float32)

    def _skin(self, mats):
        homogeneous = np.c_[self.positions.astype(np.float64), np.ones(len(self.positions))]
        out = np.zeros((len(self.positions), 3))
        for slot in range(4):
            m = mats[self.joints[:, slot]]
            out += self.weights[:, slot:slot+1] * np.einsum("nij,nj->ni", m, homogeneous)[:, :3]
        return out

    @staticmethod
    def _vertex_normals(positions, indices):
        normals = np.zeros_like(positions)
        tri = indices.reshape(-1, 3)
        face = np.cross(positions[tri[:, 1]]-positions[tri[:, 0]], positions[tri[:, 2]]-positions[tri[:, 0]])
        for k in range(3):
            np.add.at(normals, tri[:, k], face)
        return normals / np.maximum(1e-9, np.linalg.norm(normals, axis=1, keepdims=True))


# --------------------------------------------------------------------------------------
# Posing
# --------------------------------------------------------------------------------------

def _normalize(v):
    n = float(np.linalg.norm(v))
    return v/n if n > 1e-9 else v


def _palm_frame(wrist, index_mcp, middle_mcp, pinky_mcp, previous=None):
    """Build an orthonormal palm basis without flips when the hand turns edge-on.

    The projected index-to-pinky axis becomes almost zero when the palm is edge-on
    to the camera. Normalizing that near-zero vector used to collapse the basis and
    make the thumb twist abruptly. Reuse the previous across-axis in that singular
    region, project it onto the new palm plane, and choose the closest sign.
    """
    up = _normalize(np.asarray(middle_mcp, dtype=np.float64) - wrist)
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.asarray(previous[:, 0], dtype=np.float64) if previous is not None else np.array([0.0, 1.0, 0.0])
        up = _normalize(up)

    across_raw = np.asarray(index_mcp, dtype=np.float64) - pinky_mcp
    across = across_raw - up*float(np.dot(across_raw, up))
    across_len = float(np.linalg.norm(across))
    raw_len = float(np.linalg.norm(across_raw))
    if across_len < max(1e-6, raw_len*0.12):
        if previous is not None:
            previous_across = np.asarray(previous[:, 1], dtype=np.float64)
            across = previous_across - up*float(np.dot(previous_across, up))
        if float(np.linalg.norm(across)) < 1e-6:
            # Pick whichever cardinal axis is least parallel to the palm's long axis.
            basis = min((np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                         np.array([0.0, 0.0, 1.0])), key=lambda axis: abs(float(np.dot(axis, up))))
            across = basis - up*float(np.dot(basis, up))
    across = _normalize(across)
    if previous is not None:
        # Blend in the previous across axis near the edge-on singularity. This avoids
        # the sudden twist when tracked across-width changes from noisy to well-defined.
        previous_across = np.asarray(previous[:, 1], dtype=np.float64)
        previous_across = previous_across - up*float(np.dot(previous_across, up))
        previous_len = float(np.linalg.norm(previous_across))
        raw_len = max(1e-9, float(np.linalg.norm(across_raw)))
        confidence = min(1.0, max(0.0, (across_len/raw_len - 0.08)/0.34))
        if previous_len > 1e-7:
            previous_across = previous_across/previous_len
            if float(np.dot(across, previous_across)) < 0.0:
                across = -across
            across = _normalize(confidence*across + (1.0-confidence)*previous_across)
        if float(np.dot(across, previous[:, 1])) < 0.0:
            across = -across
    normal = _normalize(np.cross(up, across))
    if float(np.linalg.norm(normal)) < 1e-7:
        # Last-resort guard against malformed or momentarily collapsed landmarks.
        fallback = np.asarray(previous[:, 2], dtype=np.float64) if previous is not None else np.array([0.0, 0.0, 1.0])
        normal = _normalize(fallback - up*float(np.dot(fallback, up)))
    across = _normalize(np.cross(normal, up))
    return np.column_stack([up, across, normal])


def _align(u, v, preferred_axes=()):
    """Smallest rotation taking u onto v, robust to collapsed/noisy landmarks.

    A zero-length target direction is not evidence for a 180-degree turn. MediaPipe
    can briefly collapse adjacent finger joints when a digit is occluded; treating that
    as an antiparallel direction caused arbitrary thumb/finger flips. For true 180-degree
    turns, use palm-relative axes first so the roll remains anatomically stable.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u_len, v_len = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if (not np.isfinite(u_len) or not np.isfinite(v_len) or
            u_len < 1e-8 or v_len < 1e-8):
        return np.eye(3)
    u, v = u/u_len, v/v_len
    c = min(1.0, max(-1.0, float(np.dot(u, v))))
    axis = np.cross(u, v)
    sine = float(np.linalg.norm(axis))
    if sine < 1e-8:
        if c >= 0.0:
            return np.eye(3)
        axis = None
        candidates = list(preferred_axes or ())
        candidates.extend(([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]))
        for candidate in candidates:
            candidate = np.asarray(candidate, dtype=np.float64)
            projected = candidate - u*float(np.dot(candidate, u))
            length = float(np.linalg.norm(projected))
            if length > 1e-7:
                axis = projected/length
                break
        if axis is None:
            return np.eye(3)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2*k@k
    axis /= sine
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sine*k + (1-c)*(k@k)


def pose_joint_matrices(model: HandModel, target: np.ndarray, target_palm_frame=None) -> np.ndarray:
    """4x4 skinning matrix per joint mapping rest-mesh vertices onto the target pose.

    target: (21, 3) landmark positions in overlay pixels (x right, y down, z away).
    """
    rest = model.landmark_head
    scale = float(np.linalg.norm(target[9]-target[0])) / max(1e-9, model.palm_length)
    target_frame = (target_palm_frame if target_palm_frame is not None else
                    _palm_frame(target[0], target[5], target[9], target[17]))
    r_root = target_frame @ _palm_frame(rest[0], rest[5], rest[9], rest[17]).T
    target_palm_length = float(np.linalg.norm(target[9]-target[0]))
    # Below this fraction of the palm, a segment direction is mostly landmark noise.
    # Keep that bone aligned with the palm instead of turning it toward a random noisy
    # vector. The threshold scales with the hand, so it remains useful at 0.1x display.
    min_direction_length = max(1e-6, target_palm_length*0.025)
    rotations, heads, stretch = {}, {}, {}
    for k, canonical in enumerate(model.canonical):
        landmark = CANONICAL_LANDMARK.get(canonical) if canonical else None
        if landmark is not None and landmark in NEXT_LANDMARK and NEXT_LANDMARK[landmark] in rest:
            child = NEXT_LANDMARK[landmark]
            rest_dir = rest[child] - rest[landmark]
            target_dir = target[child]-target[landmark]
            if float(np.linalg.norm(target_dir)) < min_direction_length:
                # The joint direction has collapsed (common when MediaPipe guesses an
                # occluded fingertip). Preserve the anatomical palm-relative direction.
                rotations[k] = r_root
            else:
                rotations[k] = _align(
                    r_root @ rest_dir, target_dir,
                    preferred_axes=(target_frame[:, 2], target_frame[:, 1], target_frame[:, 0]),
                ) @ r_root
            heads[k] = target[landmark]
            # Stretch the bone along its own axis to the user's segment length, so the
            # mesh joints - and above all the fingertips - land on the tracked landmarks
            # instead of wherever the model's own finger lengths would put them.
            rest_len = float(np.linalg.norm(rest_dir))
            if rest_len > 1e-9:
                # For the last bone, the tracked tip is the end of the mesh fingertip.
                reach_len = rest_len + model.tip_cap.get(child, 0.0)
                factor = float(np.linalg.norm(target[child]-target[landmark])) / (scale*reach_len)
                # Finger segment lengths vary substantially when curled or extended. The
                # old 0.55–1.8 clamp caused tips to miss the tracked joints and made the
                # hand look as though the wrong finger was bending, especially at small
                # display scales. Keep the guardrails, but let the rig match more of the
                # observed range without flattening a curled joint.
                stretch[k] = (min(2.5, max(0.15, factor)), rest_dir/rest_len)
        elif landmark is not None and landmark != 0:
            heads[k] = None                      # fingertip: placed along its parent bone below
        else:
            rotations[k] = r_root                # wrist, metacarpals, unmapped bones
            heads[k] = target[0] + scale*(r_root @ (model.heads[k]-rest[0]))
    # Fingertips take the rotation of the distal bone.
    for k, canonical in enumerate(model.canonical):
        if k in rotations:
            continue
        landmark = CANONICAL_LANDMARK.get(canonical)
        parent_landmark = PREVIOUS_LANDMARK.get(landmark)
        parent_joint = next((j for j, c in enumerate(model.canonical)
                             if c and CANONICAL_LANDMARK.get(c) == parent_landmark), None)
        rotations[k] = rotations.get(parent_joint, r_root)
    # Fingertip joints carry their distal bone's stretch, so the very tip stays on it.
    for k, canonical in enumerate(model.canonical):
        landmark = CANONICAL_LANDMARK.get(canonical)
        if k not in stretch and landmark in PREVIOUS_LANDMARK:
            parent_joint = next((j for j, c in enumerate(model.canonical)
                                 if c and CANONICAL_LANDMARK.get(c) == PREVIOUS_LANDMARK[landmark]), None)
            if parent_joint in stretch:
                stretch[k] = stretch[parent_joint]
            if heads.get(k) is None and parent_joint is not None:
                # Carry the tip joint along the (stretched) distal bone from its parent.
                linear = scale*rotations[parent_joint]
                if parent_joint in stretch:
                    factor, axis = stretch[parent_joint]
                    linear = linear @ (np.eye(3) + (factor-1.0)*np.outer(axis, axis))
                heads[k] = heads[parent_joint] + linear @ (model.heads[k]-model.heads[parent_joint])
            elif heads.get(k) is None:
                heads[k] = target[landmark]
    # Rigs without fingertip bones: any joint still unplaced sits on its own landmark.
    for k, canonical in enumerate(model.canonical):
        if heads.get(k) is None:
            landmark = CANONICAL_LANDMARK.get(canonical)
            heads[k] = target[landmark] if landmark is not None else target[0]
    mats = np.empty((model.joint_count, 4, 4), dtype=np.float64)
    for k in range(model.joint_count):
        linear = scale*rotations[k]
        if k in stretch:
            factor, axis = stretch[k]
            linear = linear @ (np.eye(3) + (factor-1.0)*np.outer(axis, axis))
        delta = np.eye(4)
        delta[:3, :3] = linear @ model.to_screen
        delta[:3, 3] = heads[k] - linear @ model.heads[k]
        mats[k] = delta @ model.rest_joint_mats[k]
    return mats


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

_VERTEX_SHADER = """
#version 330
uniform mat4 bones[64];
uniform vec4 frame;      // left, top, width, height (pixels)
uniform vec2 depth;      // centre, half-range (pixels)
in vec3 in_position;
in vec3 in_normal;
in ivec4 in_joints;
in vec4 in_weights;
in float in_fade;
out vec3 v_normal;
out float v_fade;
void main() {
    mat4 skin = in_weights.x * bones[in_joints.x] + in_weights.y * bones[in_joints.y]
              + in_weights.z * bones[in_joints.z] + in_weights.w * bones[in_joints.w];
    vec3 p = (skin * vec4(in_position, 1.0)).xyz;
    v_normal = mat3(skin) * in_normal;
    v_fade = in_fade;
    // Screen top maps to NDC -1 so glReadPixels rows come back top-down.
    float x = (p.x - frame.x) / frame.z * 2.0 - 1.0;
    float y = (p.y - frame.y) / frame.w * 2.0 - 1.0;
    float z = clamp((p.z - depth.x) / depth.y, -1.0, 1.0);
    gl_Position = vec4(x, y, z, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform vec3 base_colour;
in vec3 v_normal;
in float v_fade;
out vec4 f_colour;
void main() {
    if (v_fade <= 0.01) discard;

    // A soft studio key, broad fill and very faint reflected light keep fingers
    // readable against both bright and dark desktop backgrounds.  This is one
    // shading pass: no extra mesh draw, shadow buffer or per-frame CPU work.
    vec3 n = normalize(v_normal);
    vec3 view = vec3(0.0, 0.0, -1.0);
    if (dot(n, view) < 0.0) n = -n; // back of an open hand is also visible
    vec3 key = normalize(vec3(-0.43, -0.63, -0.74));
    vec3 fill = normalize(vec3(0.77, 0.12, -0.62));
    vec3 bounce = normalize(vec3(-0.15, 0.84, -0.52));

    float ndv = clamp(dot(n, view), 0.0, 1.0);
    // Wrapped diffuse avoids harsh stripes at round knuckles and fingertips.
    float key_light = clamp((dot(n, key) + 0.35) / 1.35, 0.0, 1.0);
    float fill_light = max(dot(n, fill), 0.0);
    float bounce_light = max(dot(n, bounce), 0.0);
    float level = 0.44 + 0.57 * key_light + 0.16 * fill_light
                + 0.055 * bounce_light;

    // Skin is warmer in shadow; highlights are broad and satiny rather than
    // sharp and plastic.  Preserve the original user-selected skin colour.
    vec3 shadow_tint = mix(vec3(0.96, 0.89, 0.86), vec3(1.0), key_light);
    vec3 colour = base_colour * shadow_tint * level;
    vec3 half_key = normalize(key + view);
    float satin = pow(max(dot(n, half_key), 0.0), 17.0) * 0.095;
    float sheen = pow(max(dot(n, half_key), 0.0), 52.0) * 0.035;
    colour += vec3(1.0, 0.94, 0.89) * (satin + sheen);

    // A narrow, slightly shaded rim distinguishes overlapping finger edges,
    // without thick black outlines or a second geometry/rendering pass.
    float edge = pow(1.0 - ndv, 2.4);
    colour *= 1.0 - 0.105 * edge;
    colour += vec3(0.035, 0.033, 0.031) * pow(edge, 3.0);

    // The existing wrist fade and premultiplied-alpha overlay remain intact.
    f_colour = vec4(clamp(colour, 0.0, 1.0) * v_fade, v_fade);
}
"""


class HandMeshRenderer:
    """Owns the OpenGL context and per-side GPU meshes; renders posed hands to images."""

    def __init__(self, bundled_dir: Path = BUNDLED_HANDS_DIR, custom_dir: Path = CUSTOM_HANDS_DIR):
        self.bundled_dir = Path(bundled_dir)
        self.custom_dir = Path(custom_dir)
        self.error = ""
        self._ctx = None
        self._program = None
        self._models = {}
        self._vaos = {}
        self._targets = {}
        self._palm_frames = {}
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def _model_paths(self):
        custom = {side: self.custom_dir / f"custom-{side}.glb" for side in ("left", "right")}
        bundled = {side: self.bundled_dir / f"generic-hand-{side}.glb" for side in ("left", "right")}
        if custom["left"].is_file() or custom["right"].is_file():
            paths = {}
            for side, other in (("left", "right"), ("right", "left")):
                paths[side] = (custom[side], False) if custom[side].is_file() else (custom[other], True)
            return paths
        return {side: (bundled[side], False) for side in ("left", "right")}

    def _ensure(self):
        if self._ctx is not None or self._failed:
            return not self._failed
        try:
            import moderngl
        except Exception as exc:
            self._fail(f"3D hands need the 'moderngl' package ({exc}).")
            return False
        ctx = None
        errors = []
        for kwargs in ({}, {"backend": "egl"}):
            try:
                ctx = moderngl.create_standalone_context(**kwargs)
                break
            except Exception as exc:
                errors.append(f"{kwargs.get('backend', 'default')}: {exc}")
        if ctx is None:
            self._fail("No OpenGL context for 3D hands (" + "; ".join(errors) + ").")
            return False
        try:
            self._ctx = ctx
            self._program = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
            for side, (path, mirror) in self._model_paths().items():
                model = HandModel(path, mirror=mirror)
                if model.joint_count > 64:
                    raise ValueError(f"{path.name} has {model.joint_count} joints (max 64)")
                self._models[side] = model
                vertex_data = np.concatenate([model.positions, model.normals, model.fade[:, None]], axis=1).astype("f4")
                vbo = ctx.buffer(vertex_data.tobytes())
                jbo = ctx.buffer(model.joints.astype("i4").tobytes())
                wbo = ctx.buffer(model.weights.astype("f4").tobytes())
                ibo = ctx.buffer(model.indices.astype("u4").tobytes())
                self._vaos[side] = ctx.vertex_array(self._program, [
                    (vbo, "3f 3f 1f", "in_position", "in_normal", "in_fade"),
                    (jbo, "4i", "in_joints"),
                    (wbo, "4f", "in_weights"),
                ], index_buffer=ibo, index_element_size=4)
        except Exception as exc:
            self._fail(f"Could not load the 3D hand model: {type(exc).__name__}: {exc}")
            return False
        return True

    def _fail(self, message):
        self._failed = True
        self.error = message
        try:
            if self._ctx is not None:
                self._ctx.release()
        except Exception:
            pass
        self._ctx = None

    def _target_for(self, width, height):
        """Reuse offscreen buffers; grow them in 128 px steps when a hand gets bigger."""
        need_w, need_h = int(math.ceil(width/128.0)*128), int(math.ceil(height/128.0)*128)
        current = self._targets.get("size")
        if current is None or current[0] < need_w or current[1] < need_h:
            for item in self._targets.get("objects", []):
                try:
                    item.release()
                except Exception:
                    pass
            size = (max(need_w, current[0] if current else 0), max(need_h, current[1] if current else 0))
            ctx = self._ctx
            # 2x MSAA is sufficient for a small translucent hand overlay and is
            # much cheaper on software OpenGL drivers (common on desktop Linux).
            max_samples = int(getattr(ctx, "max_samples", 0) or 0)
            samples = 2 if max_samples >= 2 else 0
            colour = ctx.renderbuffer(size, 4, samples=samples)
            depth = ctx.depth_renderbuffer(size, samples=samples)
            msaa = ctx.framebuffer(colour, depth)
            resolve_colour = ctx.renderbuffer(size, 4)
            resolve = ctx.framebuffer(resolve_colour)
            self._targets = {"size": size, "msaa": msaa, "resolve": resolve,
                             "objects": [msaa, resolve, colour, depth, resolve_colour]}
        return self._targets["msaa"], self._targets["resolve"]

    def model_reach(self, side, palm_pixels):
        model = self._models.get(side)
        if model is None:
            return 0.0
        return model.reach * palm_pixels / max(1e-9, model.palm_length)

    def render(self, side: str, target: np.ndarray, colour_rgb):
        """Render one posed hand. Returns (rgba_bytes, width, height, left, top) or None."""
        if not self._ensure():
            return None
        model = self._models.get(side)
        if model is None:
            return None
        try:
            previous = self._palm_frames.get(side)
            previous_frame = previous[0] if previous and (time.monotonic()-previous[1]) < 0.35 else None
            target_frame = _palm_frame(target[0], target[5], target[9], target[17], previous=previous_frame)
            self._palm_frames[side] = (target_frame, time.monotonic())
            mats = pose_joint_matrices(model, target, target_palm_frame=target_frame)
            palm_pixels = float(np.linalg.norm(target[9]-target[0]))
            margin = self.model_reach(side, palm_pixels)*1.15 + 4.0
            left = math.floor(float(target[:, 0].min()) - margin)
            top = math.floor(float(target[:, 1].min()) - margin)
            width = int(math.ceil(float(target[:, 0].max()) + margin)) - left
            height = int(math.ceil(float(target[:, 1].max()) + margin)) - top
            if width <= 0 or height <= 0 or width*height > 16_000_000:
                return None
            msaa, resolve = self._target_for(width, height)
            ctx = self._ctx
            msaa.use()
            ctx.viewport = (0, 0, width, height)
            ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0, viewport=(0, 0, width, height))
            ctx.enable(ctx.DEPTH_TEST)
            ctx.disable(ctx.CULL_FACE)
            padded = np.zeros((64, 4, 4), dtype="f4")
            padded[:len(mats)] = mats.astype("f4")
            self._program["bones"].write(padded.transpose(0, 2, 1).tobytes())
            self._program["frame"].value = (float(left), float(top), float(width), float(height))
            z_centre = float(target[:, 2].mean())
            z_range = max(width, height, 1) * 1.5
            self._program["depth"].value = (z_centre, z_range)
            self._program["base_colour"].value = tuple(max(0.0, min(1.0, c/255.0)) for c in colour_rgb)
            self._vaos[side].render()
            ctx.copy_framebuffer(resolve, msaa)
            data = resolve.read(viewport=(0, 0, width, height), components=4, alignment=1)
            return data, width, height, left, top
        except Exception as exc:
            self._fail(f"3D hand rendering failed: {type(exc).__name__}: {exc}")
            return None
