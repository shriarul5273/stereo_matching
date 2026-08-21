import json

import pytest

from stereo_matching import cli


def test_list_models_json(capsys):
    assert cli.main(["list-models", "--json"]) == 0

    variants = json.loads(capsys.readouterr().out)
    assert "raft-stereo" in variants
    assert "foundation-stereo" in variants


def test_info_json_is_offline(capsys):
    assert cli.main(["info", "--model", "raft-stereo", "--json"]) == 0

    details = json.loads(capsys.readouterr().out)
    assert details["model"] == "raft-stereo"
    assert details["model_type"] == "raft-stereo"


def test_predict_requires_both_camera_parameters():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--model",
            "raft-stereo",
            "--left",
            "left.png",
            "--right",
            "right.png",
            "--focal-length",
            "700",
        ]
    )

    with pytest.raises(SystemExit, match="Pass both"):
        cli.cmd_predict(args)


def test_export_parser_supports_precision_and_shape():
    args = cli.build_parser().parse_args(
        [
            "export",
            "--model",
            "raft-stereo",
            "--output",
            "model.onnx",
            "--precision",
            "fp16",
            "--height",
            "384",
            "--width",
            "640",
            "--verify",
        ]
    )

    assert args.func is cli.cmd_export
    assert args.precision == "fp16"
    assert (args.height, args.width) == (384, 640)
    assert args.verify is True


def test_quantize_onnx_parser_defaults_to_verified_int8():
    args = cli.build_parser().parse_args(
        [
            "quantize-onnx",
            "--input",
            "model.onnx",
            "--output",
            "model.int8.onnx",
        ]
    )

    assert args.func is cli.cmd_quantize_onnx
    assert args.weight_type == "int8"
    assert args.no_verify is False
