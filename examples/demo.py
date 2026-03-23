"""
demo.py — Run all registered stereo models and save colored disparity maps.

Usage
-----
python demo.py
"""

import os
import time

import cv2
from PIL import Image

# -----------------------------------------------------------------------
# Configuration — edit these as needed
# -----------------------------------------------------------------------
LEFT_IMAGE  = "assets/example1/left.png"
RIGHT_IMAGE = "assets/example1/right.png"
OUTPUT_DIR  = "examples/output"
COLORMAP    = "turbo"
DEVICE      = None   # None = auto-detect (cuda > mps > cpu)

# All registered model variants to run
MODELS = [
    "raft-stereo",
    "raft-stereo-middlebury",
    "raft-stereo-eth3d",
    "raft-stereo-realtime",
    "crestereo",
    "aanet",
    "aanet-kitti2012",
    "aanet-sceneflow",
]
# -----------------------------------------------------------------------


def get_device():
    import torch
    if DEVICE is not None:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id, device):
    from stereo_matching import AutoStereoModel
    print(f"  Loading '{model_id}' ...")
    return AutoStereoModel.from_pretrained(model_id, device=device)


def run_inference(model, processor, left_img, right_img):
    import torch
    inputs  = processor(left_img, right_img)
    left_t  = inputs["left_values"].to(next(model.parameters()).device)
    right_t = inputs["right_values"].to(next(model.parameters()).device)

    t0 = time.perf_counter()
    with torch.no_grad():
        disparity_raw = model(left_t, right_t)
    elapsed = time.perf_counter() - t0

    result = processor.postprocess(
        disparity_raw,
        inputs["original_sizes"],
        colorize=True,
        colormap=COLORMAP,
    )
    return result, elapsed


def save_results(result, model_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{model_id.replace('-', '_')}_disp.png"
    path = os.path.join(OUTPUT_DIR, filename)
    colored_bgr = cv2.cvtColor(result.colored_disparity, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, colored_bgr)
    return path


def main():
    device = get_device()
    print(f"Device: {device}\n")

    left_img  = Image.open(LEFT_IMAGE).convert("RGB")
    right_img = Image.open(RIGHT_IMAGE).convert("RGB")
    print(f"Images: {LEFT_IMAGE}  ({left_img.width}×{left_img.height})\n")

    from stereo_matching.processing_utils import StereoProcessor

    for model_id in MODELS:
        print(f"[{model_id}]")
        try:
            model = load_model(model_id, device)
            processor = StereoProcessor(model.config)

            result, elapsed = run_inference(model, processor, left_img, right_img)

            path = save_results(result, model_id)

            disp = result.disparity
            print(f"  Time      : {elapsed * 1000:.1f} ms")
            print(f"  Disparity : {disp.min():.1f} – {disp.max():.1f} px  (mean {disp.mean():.1f})")
            print(f"  Saved to  : {path}\n")

            del model
        except Exception as exc:
            print(f"  ERROR: {exc}\n")


if __name__ == "__main__":
    main()
