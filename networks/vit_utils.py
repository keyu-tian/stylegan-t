# MIT License
#
# Copyright (c) 2021 Intel ISL (Intel Intelligent Systems Lab)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Based on code from https://github.com/isl-org/DPT

"""Flexible configuration and feature extraction of timm VisionTransformers."""
import os
import types
import math
from typing import Callable, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_vit(pretrained: nn.Module, x: torch.Tensor) -> dict:
    _, _, H, W = x.size()
    _ = pretrained.model.forward_flex(x)
    return {k: pretrained.rearrange(v) for k, v in activations.items()}


def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    posemb_tok, posemb_grid = (
        posemb[:, :self.start_index],
        posemb[0, self.start_index:],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
    # patch proj and dynamically resize
    B, C, H, W = x.size()
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, H // self.patch_size[1], W // self.patch_size[0]
    )

    # add cls token
    cls_tokens = self.cls_token.expand(
        x.size(0), -1, -1
    )
    x = torch.cat((cls_tokens, x), dim=1)

    # forward pass
    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)
    return x


activations = {}


def get_activation(name: str) -> Callable:
    def hook(model, input, output):
        activations[name] = output
    return hook


def make_vit_backbone(
    # model: nn.Module,
    # patch_size: List[int] = [16, 16],
    hooks: List[int] = [2, 5, 8, 11],   # 012, 345, 678, 91011
    hook_patch: bool = True,
    start_index: List[int] = 1,
):
    assert len(hooks) == 4

    patch_size = [16, 16]
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
    ckpt = r'C:\Users\16333\Desktop\PyCharm\stylegan-t\data\dino_deitsmall16_pretrain.pth'
    if not os.path.exists(ckpt):
        ckpt = '/opt/tiger/run_trial/data/dino_deitsmall16_pretrain.pth'
    missing, unexpected = model.load_state_dict(torch.load(ckpt, 'cpu'))
    assert len(missing) == len(unexpected) == 0

    pretrained = nn.Module()
    pretrained.model = model

    # add hooks
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('0'))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('1'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('3'))
    if hook_patch:
        pretrained.model.pos_drop.register_forward_hook(get_activation('4'))

    # configure readout
    def rearrange(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:] + x[:, :1]     # absorb [cls] token
        return x.transpose_(1, 2)   # BLC to BCL
        
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = patch_size

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.rearrange = types.MethodType(rearrange, pretrained)
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained
