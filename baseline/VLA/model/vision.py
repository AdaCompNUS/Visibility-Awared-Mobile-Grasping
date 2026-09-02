"""Prismatic-style dual vision backbone: DINOv2-L + SigLIP-SO400M.

Both encoders run on every image at 224x224 (patch 14 -> aligned 16x16 token
grids), features are channel-concatenated per token (1024 + 1152 = 2176).
DINOv2 interpolates its position embeddings automatically for any input size;
SigLIP needs interpolate_pos_encoding=True (384-native -> 224).

Input to `forward` is raw uint8-range RGB in [0, 1] float at the SOURCE
resolution (128x128); resizing + per-encoder normalization happen here on-GPU
so the dataset and the eval policy stay trivially consistent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# per-encoder pixel normalization (HF processor values)
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


class DualVisionBackbone(nn.Module):
    def __init__(self, dino_model: str, siglip_model: str, image_size: int = 224):
        super().__init__()
        from transformers import Dinov2Model, SiglipVisionModel

        self.image_size = image_size
        self.dino = Dinov2Model.from_pretrained(dino_model)
        self.siglip = SiglipVisionModel.from_pretrained(siglip_model)  # vision tower only
        self.embed_dim = self.dino.config.hidden_size + self.siglip.config.hidden_size  # 2176
        self.grid = image_size // 14  # 16

        self.register_buffer("dino_mean", torch.tensor(_DINO_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor(_DINO_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("siglip_mean", torch.tensor(_SIGLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("siglip_std", torch.tensor(_SIGLIP_STD).view(1, 3, 1, 1), persistent=False)

    def gradient_checkpointing_enable(self):
        self.dino.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.siglip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @property
    def num_tokens(self) -> int:
        return self.grid * self.grid  # 256

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        """rgb01: (N, 3, H, W) float in [0, 1] at source res -> (N, 256, 2176)."""
        x = F.interpolate(rgb01, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False, antialias=True)
        dt = x.dtype
        d = self.dino(
            (x - self.dino_mean.to(dt)) / self.dino_std.to(dt)
        ).last_hidden_state[:, 1:]  # drop CLS
        s = self.siglip(
            (x - self.siglip_mean.to(dt)) / self.siglip_std.to(dt),
            interpolate_pos_encoding=True,
        ).last_hidden_state
        assert d.shape[1] == s.shape[1] == self.num_tokens, (d.shape, s.shape)
        return torch.cat([d, s], dim=-1)
