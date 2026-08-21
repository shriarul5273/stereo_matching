import pytest

from stereo_matching import MODEL_REGISTRY


@pytest.mark.parametrize("variant_id", MODEL_REGISTRY.list_variants())
def test_every_variant_resolves_to_a_config(variant_id):
    config_cls = MODEL_REGISTRY.get_config_cls(variant_id)
    if hasattr(config_cls, "from_variant"):
        config = config_cls.from_variant(variant_id)
    else:
        config = config_cls()

    assert config.model_type == MODEL_REGISTRY.resolve_model_type(variant_id)


def test_unknown_variant_has_actionable_error():
    with pytest.raises(ValueError, match="Unknown model identifier"):
        MODEL_REGISTRY.resolve_model_type("not-a-stereo-model")
