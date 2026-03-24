from ...configuration_utils import BaseStereoConfig

# Map from public variant ID → internal variant key
_S2M2_VARIANT_MAP = {
    "s2m2":    "S",
    "s2m2-m":  "M",
    "s2m2-l":  "L",
    "s2m2-xl": "XL",
}

_S2M2_VARIANT_CONFIGS = {
    "S":  {"feature_channels": 128, "num_transformer": 1, "checkpoint": "CH128NTR1.pth"},
    "M":  {"feature_channels": 192, "num_transformer": 2, "checkpoint": "CH192NTR2.pth"},
    "L":  {"feature_channels": 256, "num_transformer": 3, "checkpoint": "CH256NTR3.pth"},
    "XL": {"feature_channels": 384, "num_transformer": 3, "checkpoint": "CH384NTR3.pth"},
}

_S2M2_HUB_REPO_ID = "minimok/s2m2"


class S2M2Config(BaseStereoConfig):
    model_type = "s2m2"

    def __init__(
        self,
        variant: str = "S",
        feature_channels: int = 128,
        num_transformer: int = 1,
        dim_expansion: int = 1,
        use_positivity: bool = False,
        output_upsample: bool = False,
        refine_iter: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.feature_channels = feature_channels
        self.num_transformer = num_transformer
        self.dim_expansion = dim_expansion
        self.use_positivity = use_positivity
        self.output_upsample = output_upsample
        self.refine_iter = refine_iter

    @classmethod
    def from_variant(cls, variant_id: str) -> "S2M2Config":
        if variant_id not in _S2M2_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_S2M2_VARIANT_MAP.keys())}"
            )
        internal = _S2M2_VARIANT_MAP[variant_id]
        cfg = _S2M2_VARIANT_CONFIGS[internal]
        return cls(
            variant=internal,
            feature_channels=cfg["feature_channels"],
            num_transformer=cfg["num_transformer"],
        )

    @property
    def checkpoint_filename(self) -> str:
        return _S2M2_VARIANT_CONFIGS.get(self.variant, {}).get("checkpoint", "s2m2.pth")

    @property
    def hub_repo_id(self) -> str:
        return _S2M2_HUB_REPO_ID
