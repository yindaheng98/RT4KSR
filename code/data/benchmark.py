import os
import random
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from utils.image import modcrop_tensor
from data import transforms
from data.basedataset import BaseDataset


class Benchmark(BaseDataset):
    def __init__(self,
                 dataroot: str,
                 name: str,
                 mode: str,
                 scale: int,
                 crop_size: int = 64,
                 rgb_range: int = 1) -> None:
        super(Benchmark, self).__init__(dataroot=dataroot,
                                        name=name,
                                        mode=mode,
                                        scale=scale,
                                        crop_size=crop_size,
                                        rgb_range=rgb_range)
        
        self.lr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, f"LR_bicubic_x{self.scale}")
        self.hr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, "HR")
        
        self.lr_files = [os.path.join(self.lr_dir_path, x) for x in sorted(os.listdir(self.lr_dir_path))]
        self.hr_files = [os.path.join(self.hr_dir_path, x) for x in sorted(os.listdir(self.hr_dir_path))]
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(rgb_range=self.rgb_range)
        ])
        self.degrade = transforms.BicubicDownsample(scale)

    def random_crop(self, lr, hrs):
        assert lr.shape[-2] >= self.crop_size
        assert lr.shape[-1] >= self.crop_size
        x0 = random.randint(0, lr.shape[-2]-self.crop_size)
        y0 = random.randint(0, lr.shape[-1]-self.crop_size)
        x1 = x0+self.crop_size
        y1 = y0+self.crop_size
        return lr[..., x0:x1, y0:y1], [hr[..., x0*self.scale:x1*self.scale, y0*self.scale:y1*self.scale] for hr in hrs]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_index(index)
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")

        lr = self.transforms(lr)
        hr = self.transforms(hr)
        
        # assert that images are divisable by 2
        c, lr_h, lr_w = lr.shape
        lr_hr, lr_wr = int(lr_h/2), int(lr_w/2)
        lr = lr[:, :lr_hr*2, :lr_wr*2]
        hr = hr[:, :lr.shape[-2] * self.scale, :lr.shape[-1] * self.scale]
        
        assert lr.shape[-1] * self.scale == hr.shape[-1]
        assert lr.shape[-2] * self.scale == hr.shape[-2]
        
        if self.crop_size:
            lr, (hr,) = self.random_crop(lr, (hr,))

        return {"lr":lr.to(torch.float32), "hr":hr.to(torch.float32)}


def set5(config):
    return Benchmark(config.dataroot, "Set5", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def set14(config):
    return Benchmark(config.dataroot, "Set14", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def b100(config):
    return Benchmark(config.dataroot, "B100", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def urban100(config):
    return Benchmark(config.dataroot, "Urban100", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def div2k(config):
    return Benchmark(config.dataroot, "DIV2K", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def div2k_train(config):
    return Benchmark(config.dataroot, "DIV2K", mode="train", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def nerfout(config):
    return Benchmark(config.dataroot, "nerfout", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def nerfout_train(config):
    return Benchmark(config.dataroot, "nerfout", mode="train", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)
