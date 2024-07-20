import argparse
import os
import cv2
import shutil
import re

parser = argparse.ArgumentParser()
parser.add_argument("--hrsrcroot", type=str, required=True)  # Full NeRF output
parser.add_argument("--grsrcroot", type=str, required=True)  # Gray NeRF output
parser.add_argument("--crsrcroot", type=str, required=True)  # Low-quality color NeRF output
parser.add_argument("--dataroot", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--valratio", type=int, default=4)
parser.add_argument("--match", type=str, default=r"[0-9]+.png")

if __name__ == "__main__":
    args = parser.parse_args()
    hr_train_dir_path = os.path.join(args.dataroot, "testsets", args.name, "train", "HR")
    gr_train_dir_path = os.path.join(args.dataroot, "testsets", args.name, "train", "Gray")
    cr_train_dir_path = os.path.join(args.dataroot, "testsets", args.name, "train", "LR_bicubic_x1")
    os.makedirs(hr_train_dir_path, exist_ok=True)
    os.makedirs(gr_train_dir_path, exist_ok=True)
    os.makedirs(cr_train_dir_path, exist_ok=True)
    hr_val_dir_path = os.path.join(args.dataroot, "testsets", args.name, "val", "HR")
    gr_val_dir_path = os.path.join(args.dataroot, "testsets", args.name, "val", "Gray")
    cr_val_dir_path = os.path.join(args.dataroot, "testsets", args.name, "val", "LR_bicubic_x1")
    os.makedirs(hr_val_dir_path, exist_ok=True)
    os.makedirs(gr_val_dir_path, exist_ok=True)
    os.makedirs(cr_val_dir_path, exist_ok=True)
    i = 0
    for entry in os.scandir(args.hrsrcroot):
        if not re.match(args.match, entry.name):
            continue
        hr_dir_path = hr_train_dir_path
        gr_dir_path = gr_train_dir_path
        cr_dir_path = cr_train_dir_path
        if i % args.valratio == 0:
            hr_dir_path = hr_val_dir_path
            gr_dir_path = gr_val_dir_path
            cr_dir_path = cr_val_dir_path
        hrsrc = entry.path
        grsrc = os.path.join(args.grsrcroot, os.path.relpath(hrsrc, args.hrsrcroot))
        crsrc = os.path.join(args.crsrcroot, os.path.relpath(hrsrc, args.hrsrcroot))
        ext = os.path.splitext(entry.name)[1]
        hrdst = os.path.join(hr_dir_path, str(i) + ext)
        grdst = os.path.join(gr_dir_path, str(i) + ext)
        crdst = os.path.join(cr_dir_path, str(i) + ext)
        print(hrsrc, hrdst)
        print(grsrc, grdst)
        print(crsrc, crdst)
        shutil.copyfile(hrsrc, hrdst)
        shutil.copyfile(grsrc, grdst)
        shutil.copyfile(crsrc, crdst)
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
