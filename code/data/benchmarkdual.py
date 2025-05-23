import os
from PIL import Image
from typing import Tuple

import torch
import torch.nn.functional as F

from data import transforms
from data.basedataset import BaseDataset


class BenchmarkDual(BaseDataset):
    def __init__(self,
                 dataroot: str,
                 name: str,
                 mode: str,
                 scale: int,
                 crop_size: int = 64,
                 rgb_range: int = 1,
                 gr_name="Gray") -> None:
        super(BenchmarkDual, self).__init__(dataroot=dataroot,
                                            name=name,
                                            mode=mode,
                                            scale=scale,
                                            crop_size=crop_size,
                                            rgb_range=rgb_range)

        self.lr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, f"LR_bicubic_x{self.scale}")
        self.hr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, "HR")
        self.gr_dir_path = os.path.join(dataroot, "testsets", self.name, self.mode, gr_name)

        self.lr_files = [os.path.join(self.lr_dir_path, x) for x in sorted(os.listdir(self.lr_dir_path))]
        self.hr_files = [os.path.join(self.hr_dir_path, x) for x in sorted(os.listdir(self.hr_dir_path))]
        self.gr_files = [os.path.join(self.gr_dir_path, x) for x in sorted(os.listdir(self.gr_dir_path))]

        self.transforms = transforms.Compose([
            transforms.ToTensor(rgb_range=self.rgb_range)
        ])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_index(index)
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        gr = Image.open(self.gr_files[idx]).convert('RGB')

        lr = self.transforms(lr)
        hr = self.transforms(hr)
        gr = self.transforms(gr)

        # assert that images are divisable by 2
        c, lr_h, lr_w = lr.shape
        lr_hr, lr_wr = int(lr_h/2), int(lr_w/2)
        lr = lr[:, :lr_hr*2, :lr_wr*2]
        hr = hr[:, :lr.shape[-2] * self.scale, :lr.shape[-1] * self.scale]
        gr = gr[:, :lr.shape[-2] * self.scale, :lr.shape[-1] * self.scale]

        assert lr.shape[-1] * self.scale == hr.shape[-1]
        assert lr.shape[-2] * self.scale == hr.shape[-2]
        assert gr.shape[-2] == hr.shape[-2]
        
        if self.crop_size:
            lr, (hr,gr) = self.random_crop(lr, (hr,gr))

        x = torch.cat([F.interpolate(lr.unsqueeze(0), scale_factor=self.scale,
                      mode='bicubic', align_corners=False)[0, ...], gr], dim=0)

        return {"lr": (lr.to(torch.float32), gr.to(torch.float32)), "hr": hr.to(torch.float32)}


def div2kdual(config):
    return BenchmarkDual(config.dataroot, "DIV2K", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def div2kdual_train(config):
    return BenchmarkDual(config.dataroot, "DIV2K", mode="train", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def nerfoutdual(config):
    return BenchmarkDual(config.dataroot, "nerfout", mode="val", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)


def nerfoutdual_train(config):
    return BenchmarkDual(config.dataroot, "nerfout", mode="train", scale=config.scale, crop_size=config.crop_size, rgb_range=config.rgb_range)
