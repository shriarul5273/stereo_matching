"""
stereo_matching.viz — 3D point cloud visualization of StereoOutput.

Usage::

    from stereo_matching import pipeline, viz
    import numpy as np
    from PIL import Image

    pipe = pipeline("stereo-matching", model="raft-stereo")
    result = pipe("left.png", "right.png", focal_length=721.5, baseline=0.54)

    left_img = np.array(Image.open("left.png"))

    # Interactive open3d window (requires: pip install stereo_matching[viz])
    viz.point_cloud(result, image=left_img, focal_length=721.5, baseline=0.54)

    # Matplotlib fallback — no open3d needed
    viz.point_cloud(result, image=left_img, focal_length=721.5, baseline=0.54,
                    backend="matplotlib")

    # Save to PLY — no extra dependencies
    viz.point_cloud(result, image=left_img, focal_length=721.5, baseline=0.54,
                    save_ply="scene.ply")

    # Save to GLB (glTF 2.0 binary) — no extra dependencies
    viz.point_cloud(result, image=left_img, focal_length=721.5, baseline=0.54,
                    save_glb="scene.glb")
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .output import StereoOutput


def point_cloud(
    result: StereoOutput,
    image: Optional[np.ndarray] = None,
    focal_length: Optional[float] = None,
    baseline: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    min_depth: float = 0.1,
    max_depth: float = 300.0,
    max_points: int = 500_000,
    backend: str = "open3d",
    save_ply: Optional[str] = None,
    save_glb: Optional[str] = None,
    window_name: str = "Stereo Point Cloud",
) -> None:
    """Visualize a :class:`StereoOutput` as an interactive 3D point cloud.

    Parameters
    ----------
    result:
        Output from a stereo pipeline or processor.
    image:
        Left RGB image ``(H, W, 3)`` uint8 used to colour each point.
        Falls back to ``result.colored_disparity``, then a depth-based
        turbo colourmap.
    focal_length:
        Camera focal length in pixels (fx = fy assumed).
    baseline:
        Stereo baseline in metres.
    cx, cy:
        Principal point in pixels.  Defaults to image centre.
    min_depth, max_depth:
        Depth range in metres (points outside are discarded).
        Ignored when no metric scale is available.
    max_points:
        Maximum number of points to display (random downsampling).
    backend:
        ``"open3d"`` (interactive, requires ``pip install open3d``),
        ``"matplotlib"`` (always available, slower for large clouds), or
        ``"none"`` (skip display entirely — useful when only saving files).
    save_ply:
        If given, write a binary PLY file to this path (no extra deps).
    save_glb:
        If given, write a glTF 2.0 binary (GLB) point cloud to this path
        (no extra deps).  Viewable in Blender, three.js, model-viewer, etc.
    window_name:
        Title of the open3d viewer window.
    """
    disp = result.disparity  # (H, W) float32
    H, W = disp.shape

    _cx = cx if cx is not None else W / 2.0
    _cy = cy if cy is not None else H / 2.0

    # ------------------------------------------------------------------ #
    # 1. Compute depth                                                     #
    # ------------------------------------------------------------------ #
    metric = True
    if focal_length is not None and baseline is not None:
        depth = (focal_length * baseline) / np.maximum(disp, 1e-6)
    elif result.depth is not None:
        depth = result.depth
        # We have metric depth but may not know focal_length for XY
        if focal_length is None:
            warnings.warn(
                "result.depth is available but focal_length is not provided. "
                "X and Y coordinates will be in pixel units (not metres).",
                stacklevel=2,
            )
    else:
        warnings.warn(
            "No focal_length/baseline or result.depth provided. "
            "Using 1/disparity as a pseudo-depth — no metric scale.",
            stacklevel=2,
        )
        depth = 1.0 / np.maximum(disp, 1e-6)
        metric = False

    # ------------------------------------------------------------------ #
    # 2. Build pixel grid and unproject to 3-D                            #
    # ------------------------------------------------------------------ #
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                          np.arange(H, dtype=np.float32))
    Z = depth.astype(np.float32)

    if focal_length is not None:
        X = (us - _cx) * Z / focal_length
        Y = (vs - _cy) * Z / focal_length
    else:
        # No focal length — lay out points on a pixel grid in XY
        X = us - _cx
        Y = vs - _cy

    # ------------------------------------------------------------------ #
    # 3. Mask: valid disparity + depth range                              #
    # ------------------------------------------------------------------ #
    valid = disp > 0.0
    if metric:
        valid &= (Z >= min_depth) & (Z <= max_depth)

    pts = np.stack([X, Y, Z], axis=-1)[valid]  # (N, 3)

    # ------------------------------------------------------------------ #
    # 4. Colours                                                          #
    # ------------------------------------------------------------------ #
    if image is not None:
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 2:                           # greyscale → RGB
            img = np.stack([img, img, img], axis=-1)
        colors = (img / 255.0)[valid]
    elif result.colored_disparity is not None:
        colors = (result.colored_disparity.astype(np.float32) / 255.0)[valid]
    else:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap("turbo")
        z_norm = np.clip((Z[valid] - Z[valid].min()) /
                         (Z[valid].ptp() + 1e-6), 0.0, 1.0)
        colors = cmap(z_norm)[:, :3].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 5. Downsample                                                       #
    # ------------------------------------------------------------------ #
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        colors = colors[idx]

    # ------------------------------------------------------------------ #
    # 6. Save PLY / GLB                                                   #
    # ------------------------------------------------------------------ #
    if save_ply is not None:
        _write_ply(save_ply, pts, colors)
    if save_glb is not None:
        _write_glb(save_glb, pts, colors)

    # ------------------------------------------------------------------ #
    # 7. Visualize                                                        #
    # ------------------------------------------------------------------ #
    if backend == "open3d":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required for backend='open3d'. "
                "Install it with:  pip install stereo_matching[viz]"
            ) from exc

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.visualization.draw_geometries([pcd], window_name=window_name)

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        # Swap Y↑ so the scene looks natural (Z into screen, Y up)
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
                   c=colors, s=0.3, linewidths=0)
        ax.set_xlabel("X (m)" if metric and focal_length else "X (px)")
        ax.set_ylabel("Z (depth)")
        ax.set_zlabel("Y (m)" if metric and focal_length else "Y (px)")
        ax.set_title(window_name)
        plt.tight_layout()
        plt.show()

    elif backend == "none":
        pass  # save-only mode — no display

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'open3d', 'matplotlib', or 'none'."
        )


def _write_glb(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a glTF 2.0 binary (GLB) point cloud file.

    Compatible with Blender, three.js, ``<model-viewer>``, and any glTF viewer.
    No external dependencies beyond numpy.

    Encodes positions as VEC3 float32 and colours as VEC3 float32 in [0, 1].
    Stereo points are generated in camera coordinates (X right, Y down, Z forward),
    so we convert them to a graphics-style scene basis (X right, Y up, Z back)
    before writing the GLB.
    """
    import json
    import struct

    n = len(points)
    pts_f32 = points.astype(np.float32).copy()
    pts_f32[:, 1] *= -1.0
    pts_f32[:, 2] *= -1.0
    col_f32 = colors.astype(np.float32)

    # glTF bounding box (required by spec for POSITION accessor)
    pos_min = pts_f32.min(axis=0).tolist()
    pos_max = pts_f32.max(axis=0).tolist()

    # Binary buffer: positions then colors (both float32 VEC3)
    pos_bytes = pts_f32.tobytes()
    col_bytes = col_f32.tobytes()
    bin_data = pos_bytes + col_bytes
    # Pad binary chunk to 4-byte alignment with zeros
    bin_pad = (-len(bin_data)) % 4
    bin_chunk_data = bin_data + b"\x00" * bin_pad

    pos_byte_len = len(pos_bytes)
    col_byte_len = len(col_bytes)
    stride = 12  # 3 * float32

    gltf = {
        "asset": {"version": "2.0", "generator": "stereo_matching.viz"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0, "COLOR_0": 1},
                "mode": 0,  # POINTS
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,            "byteLength": pos_byte_len, "byteStride": stride},
            {"buffer": 0, "byteOffset": pos_byte_len, "byteLength": col_byte_len, "byteStride": stride},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON chunk to 4-byte alignment with spaces
    json_pad = (-len(json_bytes)) % 4
    json_chunk_data = json_bytes + b" " * json_pad

    # Chunk headers: length (uint32) + type (uint32)
    JSON_MAGIC = 0x4E4F534A  # "JSON"
    BIN_MAGIC  = 0x004E4942  # "BIN\0"

    json_chunk = struct.pack("<II", len(json_chunk_data), JSON_MAGIC) + json_chunk_data
    bin_chunk  = struct.pack("<II", len(bin_chunk_data),  BIN_MAGIC)  + bin_chunk_data

    total_len = 12 + len(json_chunk) + len(bin_chunk)
    GLB_MAGIC   = 0x46546C67  # "glTF"
    GLB_VERSION = 2
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_len)

    with open(path, "wb") as f:
        f.write(header)
        f.write(json_chunk)
        f.write(bin_chunk)


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY file (XYZ float32 + RGB uint8).

    Compatible with open3d, CloudCompare, and MeshLab.
    No external dependencies beyond numpy.
    """
    n = len(points)
    rgb = (colors * 255).clip(0, 255).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Build interleaved vertex buffer in one numpy operation (avoids per-vertex
    # Python loop and hundreds-of-thousands of individual f.write() calls).
    pts_f32 = points.astype(np.float32)
    vertex = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = pts_f32[:, 0]
    vertex["y"] = pts_f32[:, 1]
    vertex["z"] = pts_f32[:, 2]
    vertex["red"]   = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"]  = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
