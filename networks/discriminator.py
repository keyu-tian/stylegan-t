# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from networks.shared import ResidualBlock, FullyConnectedLayer
from networks.vit_utils import make_vit_backbone, forward_vit
from training.diffaug import DiffAugment


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, dino_C: int, cond_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = dino_C
        self.c_dim = cond_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(dino_C, kernel_size=1),
            ResidualBlock(make_block(dino_C, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
        else:
            cmap_dim = 1
        self.cls = SpectralConv1d(dino_C, cmap_dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)
            c: (B, cond_dim)
        Returns:
            out: (B, 1, L)
        """
        h = self.main(x)    # BCL
        out = self.cls(h)   # (B, cmap_dim, L) or (B, 1, L)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)    # B, cond_dim => B, cmap_dim, 1
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            # (B, cmap_dim, L) * (B, cmap_dim, 1) => (B, cmap_dim, L).sum(1) => (B, 1, L)

        return out  # must be (B, 1, L)


class DINO(torch.nn.Module):
    def __init__(self, hooks: List[int] = [2,5,8,11], hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(hooks=hooks, hook_patch=hook_patch)
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features
        # _, _, H, W = x.size()
        # _ = self.model.model.forward_flex(x)
        # return {k: self.model.rearrange(v) for k, v in activations.items()}


class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        self.dino = DINO()

        heads = []
        for i in range(self.dino.n_hooks):  # 5 heads,
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
            c: (B, c_dim)
        Returns:
            logits: (B, #H * #DINO_patches i.e. L) or (B, 5*196) == (B, 980)
        """
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Transform to [0, 1].
        x = x.add(1).div(2)

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c).view(x.size(0), -1))     # B1L => BL
        logits = torch.cat(logits, dim=1)   # cat 5 BL => B, 5L

        return logits


if __name__ == '__main__':
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    with torch.no_grad():
        p = ProjectedDiscriminator(64)
        p.eval()
        x = torch.rand(2, 3, 224, 224)
        c = torch.rand(2, 64)
        print(p(x, c)[0][:8])   # 0.7690, -2.4762, -2.0082, -0.6760,  6.2214,  8.5858,  8.5007,  7.9704
