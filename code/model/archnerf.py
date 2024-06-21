import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .arch import *


class NeRFRT4KSR_Rep(RT4KSR_Rep):
    def __init__(self, num_channels_color, num_channels_gray, *args, **kwargs):
        super().__init__(*args, **kwargs, upscale=1, num_channels_in=num_channels_color + num_channels_gray)
        self.num_channels_color = num_channels_color
        self.num_channels_gray = num_channels_gray

    def forward(self, x):
        color = x[:, 0:self.num_channels_color]
        
        # Following is just copy from RT4KSR_Rep
        # stage 1
        hf = x - self.gaussian(x)

        # unshuffle to save computation
        x_unsh = self.down(x)
        hf_unsh = self.down(hf)

        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)

        # stage 2
        deep_feats = self.body(shallow_feats_lr)
        hf_feats = self.hfb(shallow_feats_hf)

        # stage 3
        if self.forget:
            deep_feats = self.tail(self.gamma * deep_feats + hf_feats)
        else:
            deep_feats = self.tail(deep_feats)

        out = self.upsample(deep_feats)
        return out + color

####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################


def nerfrt4ksr_rep(config):
    act = activation(config.act_type)
    model = NeRFRT4KSR_Rep(num_channels_color=3,
                           num_channels_gray=3,
                           num_channels_out=3,
                           num_feats=config.feature_channels,
                           num_blocks=config.num_blocks,
                           act=act,
                           eca_gamma=0,
                           forget=False,
                           is_train=config.is_train,
                           layernorm=True,
                           residual=False)
    return model
