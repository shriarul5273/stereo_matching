"""
Command-line interface for stereo_matching.

Currently supports:
  - list-models
  - info
  - predict
  - evaluate (guarded; requires the evaluation package to exist)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from .registry import MODEL_REGISTRY


def _model_source(args: argparse.Namespace) -> str:
    return args.model if getattr(args, "model", None) is not None else args.checkpoint


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    variant = getattr(args, "variant", None)
    if variant:
        kwargs["variant"] = variant
    return kwargs


def _print(msg: str = "", quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def _load_model(args: argparse.Namespace):
    from .models.auto import AutoStereoModel

    source = _model_source(args)
    kwargs = _model_kwargs(args)
    model = AutoStereoModel.from_pretrained(source, device=getattr(args, "device", None), **kwargs)
    if getattr(args, "iters", None):
        model.config.num_iters = args.iters
    return model


def _config_for_model_id(model_id: str):
    config_cls = MODEL_REGISTRY.get_config_cls(model_id)
    if hasattr(config_cls, "from_variant"):
        try:
            return config_cls.from_variant(model_id)
        except Exception:
            pass
    return config_cls()


def _save_prediction_outputs(
    result,
    left_image_path: str,
    output_dir: str,
) -> list[Path]:
    import cv2
    import numpy as np
    from PIL import Image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    disparity_u16 = np.clip(np.round(np.maximum(result.disparity, 0.0) * 256.0), 0, 65535).astype(np.uint16)
    disparity_path = out_dir / "disparity.png"
    cv2.imwrite(str(disparity_path), disparity_u16)
    saved.append(disparity_path)

    if result.colored_disparity is not None:
        color_path = out_dir / "disparity_color.png"
        color_bgr = cv2.cvtColor(result.colored_disparity, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(color_path), color_bgr)
        saved.append(color_path)

        left_rgb = np.array(Image.open(left_image_path).convert("RGB"))
        side_by_side = np.concatenate([left_rgb, result.colored_disparity], axis=1)
        side_path = out_dir / "side_by_side.png"
        cv2.imwrite(str(side_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        saved.append(side_path)

    if result.depth is not None:
        depth_path = out_dir / "depth.npy"
        np.save(str(depth_path), result.depth.astype(np.float32))
        saved.append(depth_path)

    return saved


def cmd_list_models(args: argparse.Namespace) -> int:
    variants = sorted(MODEL_REGISTRY.list_variants())
    if args.json:
        print(json.dumps(variants, indent=2))
        return 0

    print("Registered stereo model variants:")
    for variant in variants:
        print(f"  {variant}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    quiet = getattr(args, "quiet", False)

    if args.model is not None:
        config = _config_for_model_id(args.model)
        source_label = args.model
    else:
        model = _load_model(args)
        config = model.config
        source_label = args.checkpoint

    config_dict = config.to_dict()

    if args.json:
        print(json.dumps({"model": source_label, **config_dict}, indent=2))
        return 0

    _print(f"Model: {source_label}", quiet=quiet)
    for key in sorted(config_dict.keys()):
        _print(f"  {key:<15}: {config_dict[key]}", quiet=quiet)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .processing_utils import StereoProcessor

    quiet = getattr(args, "quiet", False)

    if (args.focal_length is None) ^ (args.baseline is None):
        raise SystemExit("Pass both --focal-length and --baseline together, or omit both.")

    model = _load_model(args)
    processor = StereoProcessor(model.config)

    inputs = processor(args.left, args.right)
    device = next(model.parameters()).device
    left_t = inputs["left_values"].to(device)
    right_t = inputs["right_values"].to(device)

    import torch

    with torch.no_grad():
        disparity_raw = model(left_t, right_t)

    result = processor.postprocess(
        disparity_raw,
        inputs["original_sizes"],
        colorize=True,
        colormap=args.colormap,
        focal_length=args.focal_length,
        baseline=args.baseline,
    )

    disp = result.disparity
    _print(
        f"Disparity: {disp.min():.2f} to {disp.max():.2f} px "
        f"(mean {disp.mean():.2f})",
        quiet=quiet,
    )
    if result.depth is not None:
        depth = result.depth
        _print(
            f"Depth: {depth.min():.3f} to {depth.max():.3f} m "
            f"(mean {depth.mean():.3f})",
            quiet=quiet,
        )

    if not args.no_save:
        saved = _save_prediction_outputs(result, args.left, args.output_dir)
        _print(f"Saved outputs to {Path(args.output_dir).resolve()}", quiet=quiet)
        for path in saved:
            _print(f"  {path.name}", quiet=quiet)

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        from .evaluation import evaluate as evaluate_fn
    except Exception as exc:
        raise SystemExit(
            "The evaluate command is not available in this build because "
            "`stereo_matching.evaluation` is missing."
        ) from exc

    model = _load_model(args)
    results = evaluate_fn(
        model=model,
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("Evaluation results:")
        for key in sorted(results.keys()):
            print(f"  {key}: {results[key]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stereo-matching",
        description="Stereo matching command-line interface.",
    )
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or mps.")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-essential output.")
    parser.add_argument("--verbose", action="store_true", help="Reserved for future verbose logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models", help="List all registered model variants.")
    list_parser.add_argument("--json", action="store_true", help="Print the model list as JSON.")
    list_parser.set_defaults(func=cmd_list_models)

    info_parser = subparsers.add_parser("info", help="Show configuration for a model or checkpoint.")
    info_group = info_parser.add_mutually_exclusive_group(required=True)
    info_group.add_argument("--model", help="Registered model variant ID.")
    info_group.add_argument("--checkpoint", help="Path to a local checkpoint.")
    info_parser.add_argument(
        "--variant",
        default=None,
        help="Variant hint when loading a local checkpoint.",
    )
    info_parser.add_argument("--json", action="store_true", help="Print the config as JSON.")
    info_parser.set_defaults(func=cmd_info)

    predict_parser = subparsers.add_parser("predict", help="Run stereo prediction on one image pair.")
    pred_group = predict_parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument("--model", help="Registered model variant ID.")
    pred_group.add_argument("--checkpoint", help="Path to a local checkpoint.")
    predict_parser.add_argument("--left", required=True, help="Path to the left image.")
    predict_parser.add_argument("--right", required=True, help="Path to the right image.")
    predict_parser.add_argument(
        "--variant",
        default=None,
        help="Variant hint when using --checkpoint (for example: raft-stereo or igev-stereo).",
    )
    predict_parser.add_argument("--iters", type=int, default=None, help="Override recurrent iteration count.")
    predict_parser.add_argument("--focal-length", type=float, default=None, help="Camera focal length in pixels.")
    predict_parser.add_argument("--baseline", type=float, default=None, help="Camera baseline in metres.")
    predict_parser.add_argument("--output-dir", default="./output", help="Directory to save outputs.")
    predict_parser.add_argument("--colormap", default="turbo", help="Matplotlib colormap name.")
    predict_parser.add_argument("--no-save", action="store_true", help="Print stats only; do not write files.")
    predict_parser.set_defaults(func=cmd_predict)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model on a dataset.")
    eval_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--model", help="Registered model variant ID.")
    eval_group.add_argument("--checkpoint", help="Path to a local checkpoint.")
    eval_parser.add_argument("--variant", default=None, help="Variant hint when using --checkpoint.")
    eval_parser.add_argument("--dataset", required=True, help="Dataset name.")
    eval_parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    eval_parser.add_argument("--split", default="val", help="Dataset split.")
    eval_parser.add_argument("--iters", type=int, default=None, help="Override recurrent iteration count.")
    eval_parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    eval_parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
