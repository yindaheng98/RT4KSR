import argparse
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--scale", type=int, default=2)

if __name__ == "__main__":
    args = parser.parse_args()
    hr_dir_path = os.path.join(args.dataroot, "testsets", args.name, args.mode, "HR")
    gr_dir_path = os.path.join(args.dataroot, "testsets", args.name, args.mode, "Gray")
    os.makedirs(gr_dir_path, exist_ok=True)
    for x in sorted(os.listdir(hr_dir_path)):
        hr_file = os.path.join(hr_dir_path, x)
        gr_file = os.path.join(gr_dir_path, x)
        hr = cv2.imread(hr_file)
        gr = cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(gr_file, gr)
        
