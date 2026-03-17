"""
BaseStereoModel — Base model class for all stereo matching models.

Provides weight loading, device management, and inference context.
Subclasses implement forward() with their specific network architecture.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configuration_utils import BaseStereoConfig

logger = logging.getLogger(__name__)


class BaseStereoModel(nn.Module):
    """Base wrapper around stereo matching neural networks.

    Subclasses must implement:
        - ``forward(left, right) → torch.Tensor``  (raw disparity tensor)
        - ``_load_pretrained_weights(model_id, device)`` (load checkpoint)

    Provides:
        - ``from_pretrained(model_id)`` for weight loading.
        - Device management via ``to(device)``, ``half()``, ``float()``.
        - Automatic ``torch.no_grad()`` context during inference.
        - Training helpers: ``freeze_backbone()``, ``unfreeze_backbone()``,
          ``get_parameter_groups()``, ``unfreeze_top_k_backbone_layers()``.
    """

    config_class: type = BaseStereoConfig

    def __init__(self, config: BaseStereoConfig):
        super().__init__()
        self.config = config

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            left: Left image tensor of shape (B, 3, H, W), normalized.
            right: Right image tensor of shape (B, 3, H, W), normalized.

        Returns:
            Disparity tensor of shape (B, H, W) during inference,
            or List[Tensor(B, H, W)] during training (recurrent models).
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: Optional[str] = None,
        for_training: bool = False,
        **kwargs: Any,
    ) -> "BaseStereoModel":
        """Load a pretrained model.

        Args:
            model_id: Model identifier string (e.g. "raft-stereo")
                or a local path to a checkpoint.
            device: Device to load the model to. Auto-detected if None.
            for_training: If True, keep the model in training mode after
                loading (skip the default ``.eval()`` call). Default False.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            Instantiated model with pretrained weights.
            In eval mode by default; in train mode when ``for_training=True``.
        """
        if device is None:
            device = _auto_detect_device()

        model = cls._load_pretrained_weights(model_id, device=device, **kwargs)
        model = model.to(device)
        if not for_training:
            model.eval()
        logger.info(
            f"Loaded {cls.__name__} from '{model_id}' on {device} "
            f"({'train' if for_training else 'eval'} mode)"
        )
        return model

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "BaseStereoModel":
        """Load weights from a pretrained checkpoint.

        Subclasses must override this to implement their specific loading logic
        (HuggingFace Hub, local .pth, etc.).
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _load_pretrained_weights()"
        )

    def predict(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Run inference, suppressing gradients only when not training.

        Args:
            left: Left image tensor of shape (B, 3, H, W).
            right: Right image tensor of shape (B, 3, H, W).

        Returns:
            Disparity tensor of shape (B, H, W).
        """
        if self.training:
            return self.forward(left, right)
        with torch.no_grad():
            return self.forward(left, right)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _backbone_module(self) -> Optional[nn.Module]:
        """Return the backbone sub-module, or None if not found.

        Searches common attribute names. Subclasses should override this
        to return the correct backbone when the default search is wrong.

        Returns:
            The backbone ``nn.Module``, or ``None`` if not identifiable.
        """
        for attr in ("pretrained", "encoder", "backbone", "fnet"):
            candidate = getattr(self, attr, None)
            if isinstance(candidate, nn.Module):
                return candidate
        return None

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (train decoder/head only).

        Raises:
            RuntimeError: If the backbone module cannot be identified.
        """
        backbone = self._backbone_module()
        if backbone is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no identifiable backbone module. "
                "Override _backbone_module() to specify the backbone."
            )
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info(f"Backbone frozen. Trainable params: {self._count_trainable():,}")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for full fine-tuning.

        Raises:
            RuntimeError: If the backbone module cannot be identified.
        """
        backbone = self._backbone_module()
        if backbone is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no identifiable backbone module. "
                "Override _backbone_module() to specify the backbone."
            )
        for param in backbone.parameters():
            param.requires_grad = True
        logger.info(f"Backbone unfrozen. Trainable params: {self._count_trainable():,}")

    def _count_trainable(self) -> int:
        """Return the number of trainable (requires_grad) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_groups(self, backbone_lr_scale: float = 0.1) -> List[dict]:
        """Return parameter groups with differential learning rates.

        The backbone group uses ``lr * backbone_lr_scale``.
        The rest uses the base lr (scale = 1.0).

        Args:
            backbone_lr_scale: Multiplier applied to the backbone LR.
                Default 0.1 (backbone trains 10× slower than the rest).

        Returns:
            List of two dicts: ``[{"params": other_params, "lr_scale": 1.0},
            {"params": backbone_params, "lr_scale": backbone_lr_scale}]``.
        """
        backbone = self._backbone_module()
        backbone_ids = {id(p) for p in backbone.parameters()} if backbone else set()

        other_params = [p for p in self.parameters() if id(p) not in backbone_ids and p.requires_grad]
        backbone_params = [p for p in self.parameters() if id(p) in backbone_ids and p.requires_grad]

        return [
            {"params": other_params, "lr_scale": 1.0},
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
        ]

    def unfreeze_top_k_backbone_layers(self, k: int) -> None:
        """Unfreeze the last k layers of the backbone.

        Default implementation searches for a ``.blocks`` attribute on the backbone.
        Subclasses with different backbone architectures should override this.

        Args:
            k: Number of blocks to unfreeze from the top (closest to output).
        """
        backbone = self._backbone_module()
        if backbone is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no identifiable backbone module."
            )
        if not hasattr(backbone, "blocks"):
            raise RuntimeError(
                f"{self.__class__.__name__} backbone has no .blocks attribute. "
                "Override unfreeze_top_k_backbone_layers() for this model."
            )
        all_blocks = list(backbone.blocks)
        for block in all_blocks[-k:]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(
            f"Unfroze top {k} backbone blocks. Trainable params: {self._count_trainable():,}"
        )


def _auto_detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
