"""
compare_demo.py — Gradio app comparing stereo matching models with a synchronized 3D viewer.

Runs two selected models on a stereo pair and displays the resulting point clouds
side-by-side in a synchronized GLB viewer using gradio_sync3dcompare.

Usage
-----
    cd <repo-root>
    python examples/compare_demo.py

Requirements
------------
    pip install gradio gradio_sync3dcompare
    pip install stereo_matching[viz]   # for open3d (optional)
"""

import gc
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from gradio_sync3dcompare import Sync3DCompare

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All registered model variant IDs
# ---------------------------------------------------------------------------
ALL_MODELS = [
    "raft-stereo",
    "raft-stereo-middlebury",
    "raft-stereo-eth3d",
    "raft-stereo-realtime",
    "crestereo",
    "aanet",
    "aanet-kitti2012",
    "aanet-sceneflow",
    "foundation-stereo",
    "foundation-stereo-large",
    "unimatch",
    "unimatch-mixdata",
    "unimatch-sceneflow",
    "unimatch-kitti15",
    "unimatch-middlebury",
    "igev-plusplus",
    "igev-plusplus-kitti2012",
    "igev-plusplus-kitti2015",
    "igev-plusplus-middlebury",
    "igev-plusplus-eth3d",
    "igev-stereo",
    "igev-stereo-sceneflow",
    "igev-stereo-kitti2012",
    "igev-stereo-kitti2015",
    "igev-stereo-middlebury",
    "igev-stereo-eth3d",
]

COLORMAP = "turbo"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_k_txt(k_txt_path: str) -> tuple[float, float, float, float]:
    """Parse a K.txt file and return (fx, cx, cy, baseline).

    Supports two formats:

    **Row-major 3×3 + baseline (metres)**::

        fx  0  cx  0  fy  cy  0  0  1
        0.063

    **Middlebury-style**::

        cam0=[fx 0 cx; 0 fy cy; 0 0 1]
        baseline=536.62   # millimetres
    """
    import re
    text = Path(k_txt_path).read_text()
    lines = text.strip().splitlines()

    # Middlebury format: contains "cam0=[..."
    if any("cam0=" in ln for ln in lines):
        cam0_line = next(ln for ln in lines if "cam0=" in ln)
        # Extract only the numbers inside the brackets to avoid matching
        # the '0' in 'cam0=' which would shift all indices.
        bracket_content = cam0_line.split("[", 1)[1].split("]")[0]
        nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+", bracket_content)))
        fx, cx, cy = nums[0], nums[2], nums[5]
        baseline_line = next((ln for ln in lines if ln.startswith("baseline=")), None)
        if baseline_line is None:
            raise ValueError("No 'baseline=' entry found in Middlebury K.txt")
        baseline_mm = float(baseline_line.split("=")[1])
        baseline = baseline_mm / 1000.0  # mm → metres
        return fx, cx, cy, baseline

    # Row-major 9-float format
    vals = list(map(float, lines[0].split()))
    fx, cx, cy = vals[0], vals[2], vals[5]
    baseline = float(lines[1].strip())
    return fx, cx, cy, baseline


def _run_model(
    model_id: str,
    left_pil: Image.Image,
    right_pil: Image.Image,
    focal_length: float,
    baseline: float,
    cx: Optional[float],
    cy: Optional[float],
    glb_path: str,
) -> str:
    """Load model, run inference, export GLB, return status string."""
    from stereo_matching import AutoStereoModel, viz
    from stereo_matching.processing_utils import StereoProcessor
    import torch

    device = _get_device()
    log.info("Loading model '%s' on %s ...", model_id, device)

    model = AutoStereoModel.from_pretrained(model_id, device=device)
    processor = StereoProcessor(model.config)

    inputs = processor(left_pil, right_pil)
    left_t  = inputs["left_values"].to(device)
    right_t = inputs["right_values"].to(device)

    with torch.no_grad():
        disparity_raw = model(left_t, right_t)

    result = processor.postprocess(
        disparity_raw,
        inputs["original_sizes"],
        colorize=True,
        colormap=COLORMAP,
        focal_length=focal_length,
        baseline=baseline,
    )

    left_np = np.array(left_pil)

    viz.point_cloud(
        result,
        image=left_np,
        focal_length=focal_length,
        baseline=baseline,
        cx=cx,
        cy=cy,
        backend="none",
        save_glb=glb_path,
    )

    disp = result.disparity
    status = (
        f"{model_id}: disparity {disp.min():.1f}–{disp.max():.1f} px "
        f"(mean {disp.mean():.1f})"
    )

    # Free GPU memory
    del model, disparity_raw, left_t, right_t
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return status


def _load_example_k(example_dir: Path) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Try to parse K.txt from an example directory."""
    k_path = example_dir / "K.txt"
    if k_path.exists():
        fx, cx, cy, baseline = _parse_k_txt(str(k_path))
        return fx, cx, cy, baseline
    return None, None, None, None


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def run_comparison(
    left_image,
    right_image,
    model_a: str,
    model_b: str,
    focal_length: float,
    baseline: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[dict], str]:
    """Run two models and return Sync3DCompare value + status markdown."""

    if left_image is None or right_image is None:
        return [], "Please upload both left and right images."

    if model_a == model_b:
        return [], "Please select two **different** models."

    if focal_length <= 0 or baseline <= 0:
        return [], "Please provide valid focal length and baseline (both > 0)."

    left_pil  = Image.fromarray(left_image).convert("RGB")
    right_pil = Image.fromarray(right_image).convert("RGB")

    results = []
    status_lines = []

    for i, model_id in enumerate([model_a, model_b], start=1):
        progress((i - 1) / 2, desc=f"Running {model_id} ...")
        glb_fd, glb_path = tempfile.mkstemp(suffix=".glb", prefix=f"{model_id.replace('-', '_')}_")
        os.close(glb_fd)

        try:
            status = _run_model(
                model_id, left_pil, right_pil,
                focal_length, baseline,
                cx=None, cy=None,
                glb_path=glb_path,
            )
            results.append({"name": model_id, "path": glb_path})
            status_lines.append(f"✅ {status}")
            log.info("GLB saved: %s", glb_path)
        except Exception as exc:
            log.exception("Model %s failed", model_id)
            status_lines.append(f"❌ {model_id}: {exc}")

    progress(1.0, desc="Done!")

    if len(results) < 2:
        return [], "\n\n".join(status_lines)

    viewer_value = [
        {"name": results[0]["name"], "path": results[0]["path"]},
        {"name": results[1]["name"], "path": results[1]["path"]},
    ]

    status_md = "\n\n".join(status_lines)
    return viewer_value, status_md


def _label_disparity(colored: np.ndarray, model_id: str) -> np.ndarray:
    """Add model name label to the top of a colored disparity image."""
    h, w = colored.shape[:2]

    labeled_img = np.zeros((h + 40, w, 3), dtype=np.uint8)
    labeled_img.fill(255)  # white background for label area
    labeled_img[40:, :] = colored

    font          = cv2.FONT_HERSHEY_SIMPLEX
    font_scale    = 0.7
    font_thickness = 2

    text_size = cv2.getTextSize(model_id, font, font_scale, font_thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 28

    cv2.putText(labeled_img, model_id, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return labeled_img


def _make_runtime_plot(model_ids: list[str], times_ms: list[float]):
    """Return a matplotlib Figure bar chart comparing inference times."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.barh(model_ids, times_ms, color=colors, height=0.4)

    for bar, ms in zip(bars, times_ms):
        ax.text(
            bar.get_width() + max(times_ms) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{ms:.0f} ms",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Inference time (ms)")
    ax.set_title("Runtime comparison")
    ax.set_xlim(0, max(times_ms) * 1.25)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def run_disparity_comparison(
    left_image,
    right_image,
    model_a: str,
    model_b: str,
    focal_length: float,
    baseline: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Optional[tuple], str, Optional[object]]:
    """Run two models and return labeled disparity images, status, and runtime plot."""
    from stereo_matching import AutoStereoModel
    from stereo_matching.processing_utils import StereoProcessor
    import torch

    if left_image is None or right_image is None:
        return None, "Please upload both left and right images.", None

    if model_a == model_b:
        return None, "Please select two **different** models.", None

    left_pil  = Image.fromarray(left_image).convert("RGB")
    right_pil = Image.fromarray(right_image).convert("RGB")

    disparities  = []
    status_lines = []
    runtimes_ms  = []
    device = _get_device()

    for i, model_id in enumerate([model_a, model_b], start=1):
        progress((i - 1) / 2, desc=f"Running {model_id} ...")
        try:
            model     = AutoStereoModel.from_pretrained(model_id, device=device)
            processor = StereoProcessor(model.config)

            inputs  = processor(left_pil, right_pil)
            left_t  = inputs["left_values"].to(device)
            right_t = inputs["right_values"].to(device)

            # Time only the forward pass
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                disparity_raw = model(left_t, right_t)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            runtimes_ms.append(elapsed_ms)

            result = processor.postprocess(
                disparity_raw,
                inputs["original_sizes"],
                colorize=True,
                colormap=COLORMAP,
            )

            labeled = _label_disparity(result.colored_disparity, model_id)
            disparities.append(labeled)

            disp = result.disparity
            status_lines.append(
                f"✅ **{model_id}** — {elapsed_ms:.0f} ms | "
                f"disparity {disp.min():.1f}–{disp.max():.1f} px (mean {disp.mean():.1f})"
            )

            del model, disparity_raw, left_t, right_t
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as exc:
            log.exception("Model %s failed", model_id)
            status_lines.append(f"❌ {model_id}: {exc}")
            runtimes_ms.append(0.0)

    progress(1.0, desc="Done!")

    if len(disparities) < 2:
        return None, "\n\n".join(status_lines), None

    runtime_fig = _make_runtime_plot([model_a, model_b], runtimes_ms)
    return (disparities[0], disparities[1]), "\n\n".join(status_lines), runtime_fig


def load_example(example_dir_name: str) -> tuple:
    """Load left/right images and camera params from an assets example."""
    ex_dir = ASSETS_DIR / example_dir_name
    left_np  = np.array(Image.open(ex_dir / "left.png").convert("RGB"))
    right_np = np.array(Image.open(ex_dir / "right.png").convert("RGB"))

    fx, _cx, _cy, baseline = _load_example_k(ex_dir)
    focal  = fx      if fx       is not None else 500.0
    baseln = baseline if baseline is not None else 0.1

    return left_np, right_np, focal, baseln



# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    example_dirs = sorted(
        d.name for d in ASSETS_DIR.iterdir() if d.is_dir()
    ) if ASSETS_DIR.exists() else []

    with gr.Blocks(title="Stereo Matching 3D Comparison") as app:

        gr.Markdown(
            """
# Stereo Matching — 3D Model Comparison

Select two models, upload a **rectified** stereo pair (or pick an example),
provide camera intrinsics, and click **Compare** to see synchronized 3D point
clouds side-by-side.

> **Tip:** Use the slider / orbit controls in the viewer to inspect depth quality.
"""
        )

        with gr.Accordion("ℹ️ Model Weights Notes", open=False):
            gr.Markdown(
                """
`foundation-stereo` and `foundation-stereo-large` weights are **not**
bundled and must be downloaded separately from Google Drive before use.

`foundation-stereo` and `foundation-stereo-large` use a separate Google Drive source:

**Google Drive:** https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf

Download the relevant folder (`11-33-40` for standard, `23-51-11` for large) and either:

- **Auto-cache** — install `gdown` (`pip install gdown`) and the app will download automatically on first use.
- **Manual** — place the checkpoint at `~/.cache/foundation_stereo/<folder>/model_best_bp2.pth`,
  or load directly with `FoundationStereoModel.from_pretrained("/path/to/model_best_bp2.pth", variant="standard")`.

`igev-stereo` now auto-downloads from Hugging Face:

- Repo: `shriarul5273/IGEV-Stereo`
- Files used by the registered variants:
  `sceneflow/sceneflow.pth`, `kitti/kitti12.pth`, `kitti/kitti15.pth`,
  `middlebury/middlebury.pth`, `eth3d/eth3d.pth`

You can still override it by setting `IGEV_STEREO_CHECKPOINT=/path/to/sceneflow.pth`
or by placing the checkpoint under one of the usual local cache paths.

`igev-plusplus` and `igev-plusplus-kitti2015` now auto-download from Hugging Face:

- Repo: `shriarul5273/IGEV-plusplus-Stereo`
- Files used by the registered variants:
  `sceneflow.pth`, `kitti2012.pth`, `kitti2015.pth`, `middlebury.pth`, `eth3d.pth`

You can still override them with `IGEV_PLUSPLUS_CHECKPOINT=/path/to/sceneflow.pth`
or `/path/to/kitti2015.pth`, or by placing the checkpoint under the usual local cache paths.

The remaining models download automatically from HuggingFace Hub.
"""
            )

        with gr.Row():
            # ---- Left column: inputs ----------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### Images")
                left_img  = gr.Image(label="Left image",  type="numpy")
                right_img = gr.Image(label="Right image", type="numpy")

                gr.Markdown("### Camera intrinsics")
                with gr.Row():
                    focal_input    = gr.Number(label="Focal length (px)",  value=721.5, precision=2)
                    baseline_input = gr.Number(label="Baseline (m)",        value=0.54,  precision=4)

            # ---- Right column: outputs ---------------------------------------
            with gr.Column(scale=2):
                with gr.Row():
                    model_a = gr.Dropdown(ALL_MODELS, value=ALL_MODELS[0], label="Model A")
                    model_b = gr.Dropdown(ALL_MODELS, value=ALL_MODELS[4], label="Model B")

                with gr.Tabs():
                    with gr.Tab("3D Point Cloud"):
                        viewer = Sync3DCompare(
                            value=[],
                            label="3D Point Cloud Comparison (synchronized camera)",
                            render_mode="points",
                            sync_camera=True,
                            point_size=0.5,
                            default_zoom=1,
                            min_zoom=0.5,
                            max_zoom=16.0,
                            height=520,
                            num_views=2,
                        )
                        compare_3d_btn  = gr.Button("Compare 3D", variant="primary")
                        status_3d_md    = gr.Markdown()

                    with gr.Tab("Disparity"):
                        disp_slider = gr.ImageSlider(
                            label="Disparity comparison — drag to reveal",
                            type="numpy",
                        )
                        compare_disp_btn = gr.Button("Compare Disparity", variant="primary")
                        status_disp_md   = gr.Markdown()
                        runtime_plot     = gr.Plot(label="Runtime comparison")

                # keep old name alias so wire-up below still works
                compare_btn = compare_3d_btn
                status_md   = status_3d_md

        # ---- Examples --------------------------------------------------------
        if example_dirs:
            example_rows = []
            for ex_name in example_dirs:
                ex_dir = ASSETS_DIR / ex_name
                fx, _cx, _cy, baseline = _load_example_k(ex_dir)
                example_rows.append([
                    str(ex_dir / "left.png"),
                    str(ex_dir / "right.png"),
                    fx if fx is not None else 721.5,
                    baseline if baseline is not None else 0.54,
                ])

            gr.Examples(
                examples=example_rows,
                inputs=[left_img, right_img, focal_input, baseline_input],
                label="Examples",
                examples_per_page=5,
            )

        # ---- Wire-ups --------------------------------------------------------
        compare_btn.click(
            fn=run_comparison,
            inputs=[left_img, right_img, model_a, model_b, focal_input, baseline_input],
            outputs=[viewer, status_md],
            show_progress=True,
        )

        compare_disp_btn.click(
            fn=run_disparity_comparison,
            inputs=[left_img, right_img, model_a, model_b, focal_input, baseline_input],
            outputs=[disp_slider, status_disp_md, runtime_plot],
            show_progress=True,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = create_app()
    app.launch(show_error=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
