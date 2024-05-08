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

####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################


def nerfrt4ksr_rep(config):
    act = activation(config.act_type)
    model = NeRFRT4KSR_Rep(num_channels_color=3,
                           num_channels_gray=1,
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
