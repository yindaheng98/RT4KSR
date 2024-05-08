import argparse
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2
import glob
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--hrsrcroot", type=str, required=True)  # Full NeRF output
parser.add_argument("--grsrcroot", type=str, required=True)  # Gray NeRF output
parser.add_argument("--dataroot", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--mode", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    hr_dir_path = os.path.join(args.dataroot, "testsets", args.name, args.mode, "HR")
    gr_dir_path = os.path.join(args.dataroot, "testsets", args.name, args.mode, "Gray")
    i = 0
    for path in glob.glob(os.path.join(args.hrsrcroot, "camera*")):
        for entry in os.scandir(path):
            hrsrc = entry.path
            grsrc = os.path.join(args.grsrcroot, os.path.relpath(hrsrc, args.hrsrcroot))
            ext = os.path.splitext(entry.name)[1]
            hrdst = os.path.join(hr_dir_path, str(i) + ext)
            grdst = os.path.join(gr_dir_path, str(i) + ext)
            print(hrsrc, hrdst)
            print(grsrc, grdst)
            for scale in [2, 3, 4]:
                lr_dir_path = os.path.join(args.dataroot, "testsets", args.name, args.mode, f"LR_bicubic_x{scale}")
                lrdst = os.path.join(lr_dir_path, str(i) + ext)
                print(hrsrc, lrdst)
            print("------")
