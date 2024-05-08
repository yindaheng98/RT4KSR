import os
import argparse
import torch
import pathlib


from tqdm import tqdm
from collections import OrderedDict

import data
import model
from utils import image, metrics, parser


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataroot", type=str, default=os.path.join(pathlib.Path.home(), "datasets/image_restoration"))
    parser.add_argument("--benchmark", type=str, nargs="+", default=["ntire23rtsr"])
    parser.add_argument("--checkpoints-root", type=str, default="code/checkpoints")
    parser.add_argument("--checkpoint-id", type=str, default="rt4ksr_x2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # model definitions
    parser.add_argument("--arch", type=str, default="rt4ksr_rep")
    parser.add_argument("--feature-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--act-type", type=str, default="gelu", choices=["relu", "lrelu", "gelu"])
    parser.add_argument("--is-train", action="store_true", help="Switch between training and inference mode for reparameterizable blocks.")
    parser.add_argument("--rep", action="store_true", help="Run inference with reparameterized version.")
    parser.add_argument("--save-rep-checkpoint", action="store_true", help="Save checkpoint of reparameterized model intance.")

    # data
    parser.add_argument("--scale", type=int, default=2, choices=[2,3])
    parser.add_argument("--rgb-range", type=float, default=1.0)
    
    return parser.parse_args()


def train(config):
    """
    SETUP METRICS
    """
    test_results = OrderedDict()
    test_results["psnr_rgb"] = []
    test_results["psnr_y"] = []
    test_results["ssim_rgb"] = []
    test_results["ssim_y"] = []
    test_results = test_results

    """
    SETUP MODEL
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(
        model.__dict__[config.arch](config)
    ).to(device)
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    for benchmark in config.benchmark:
        test_loader = torch.utils.data.DataLoader(
            data.__dict__[benchmark](config),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )
        print("Training...")
        pbar = tqdm(test_loader)
        for batch in pbar:
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # run method
            out = net(lr_img)
            
            # Compute the loss and its gradients
            loss = loss_fn(out, hr_img)
            loss.backward()
            pbar.set_description(f"loss={loss.item()}")

            # Adjust learning weights
            optimizer.step()


if __name__ == "__main__":
    args = train_parser()

    train(args)
