from importlib.metadata import version as installed_version

import stereo_matching


def test_version_matches_installed_metadata():
    assert stereo_matching.__version__ == installed_version("stereo_matching")


def test_public_registry_is_populated():
    assert stereo_matching.MODEL_REGISTRY.list_model_types()
    assert stereo_matching.MODEL_REGISTRY.list_variants()


def test_export_and_quantization_are_public():
    assert callable(stereo_matching.export_onnx)
    assert callable(stereo_matching.quantize_model)
    assert callable(stereo_matching.quantize_onnx)
