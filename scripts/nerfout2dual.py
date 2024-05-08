import argparse
import numpy as np
import os
import cv2
import glob
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--hrsrcroot", type=str, required=True)  # Full NeRF output
parser.add_argument("--grsrcroot", type=str, required=True)  # Gray NeRF output
parser.add_argument("--dataroot", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--valratio", type=int, default=4)

if __name__ == "__main__":
    args = parser.parse_args()
    hr_train_dir_path = os.path.join(args.dataroot, "testsets", args.name, "train", "HR")
    gr_train_dir_path = os.path.join(args.dataroot, "testsets", args.name, "train", "Gray")
    os.makedirs(hr_train_dir_path, exist_ok=True)
    os.makedirs(gr_train_dir_path, exist_ok=True)
    hr_val_dir_path = os.path.join(args.dataroot, "testsets", args.name, "val", "HR")
    gr_val_dir_path = os.path.join(args.dataroot, "testsets", args.name, "val", "Gray")
    os.makedirs(hr_val_dir_path, exist_ok=True)
    os.makedirs(gr_val_dir_path, exist_ok=True)
    i = 0
    for path in glob.glob(os.path.join(args.hrsrcroot, "camera*")):
        for entry in os.scandir(path):
            hr_dir_path = hr_train_dir_path
            gr_dir_path = gr_train_dir_path
            if i % args.valratio == 0:
                hr_dir_path = hr_val_dir_path
                gr_dir_path = gr_val_dir_path
            hrsrc = entry.path
            grsrc = os.path.join(args.grsrcroot, os.path.relpath(hrsrc, args.hrsrcroot))
            ext = os.path.splitext(entry.name)[1]
            hrdst = os.path.join(hr_dir_path, str(i) + ext)
            grdst = os.path.join(gr_dir_path, str(i) + ext)
            print(hrsrc, hrdst)
            print(grsrc, grdst)
            shutil.copyfile(hrsrc, hrdst)
            shutil.copyfile(grsrc, grdst)
            hr = cv2.imread(hrsrc)
            for scale in [2, 3, 4]:
                lr_dir_path = os.path.join(args.dataroot, "testsets", args.name, "train", f"LR_bicubic_x{scale}")
                if i % args.valratio == 0:
                    lr_dir_path = os.path.join(args.dataroot, "testsets", args.name, "val", f"LR_bicubic_x{scale}")
                os.makedirs(lr_dir_path, exist_ok=True)
                lrdst = os.path.join(lr_dir_path, str(i) + ext)
                print(hrsrc, lrdst)
                lr = cv2.resize(hr, dsize=(hr.shape[1]//scale, hr.shape[0]//scale), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(lrdst, lr)
            print("------")
            i += 1
