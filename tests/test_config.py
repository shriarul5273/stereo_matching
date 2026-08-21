from stereo_matching.configuration_utils import BaseStereoConfig


def test_config_round_trip(tmp_path):
    config = BaseStereoConfig(
        input_size=256,
        max_disparity=128,
        custom_setting="kept",
    )

    config.save_pretrained(str(tmp_path))
    loaded = BaseStereoConfig.from_pretrained(str(tmp_path))

    assert loaded == config


def test_config_dict_is_independent_copy():
    config = BaseStereoConfig()
    values = config.to_dict()
    values["mean"][0] = 0.0

    assert config.mean[0] == 0.485
