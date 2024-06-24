import os
import argparse
import torch
import torch.nn.functional as F
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
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    # model definitions
    parser.add_argument("--arch", type=str, default="rt4ksr_rep")
    parser.add_argument("--feature-channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--act-type", type=str, default="gelu", choices=["relu", "lrelu", "gelu"])
    parser.add_argument("--rep", action="store_true", help="Run inference with reparameterized version.")
    parser.add_argument("--save-rep-checkpoint", action="store_true",
                        help="Save checkpoint of reparameterized model intance.")

    # data
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--rgb-range", type=float, default=1.0)

    return parser.parse_args()


def reparameterize(config, net, device, save_rep_checkpoint=False):
    config.is_train = False
    rep_model = torch.nn.DataParallel(model.__dict__[config.arch](config)).to(device)
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = net.state_dict()

    for k, v in rep_state_dict.items():
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]

            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]

            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0

            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0

            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()

        elif "rep_conv.bias" in k:
            pass

        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    if config.checkpoint_id:
        checkpoint = dict(state_dict=rep_state_dict)
        checkpoint_path = os.path.join("code/checkpoints", config.checkpoint_id + "_rep_model.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    return rep_model


def train(config):
    config.__setattr__("is_train", True)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, weight_decay=0, betas=[0.9, 0.99])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch, eta_min=1e-6)
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
        for epoch in range(config.epoch):
            print("Training...")
            pbar = tqdm(test_loader)
            for batch in pbar:
                lr_img = [t.to(device) for t in batch["lr"]] if isinstance(batch["lr"], list) else batch["lr"].to(device)
                hr_img = batch["hr"].to(device)
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # run method
                out = net(lr_img)

                # Compute the loss and its gradients
                loss = loss_fn(out, hr_img)
                loss.backward()
                pbar.set_description(f"epoch {epoch} loss=%.6f" % loss.item())

                # Adjust learning weights
                optimizer.step()
            scheduler.step()
    if config.checkpoint_id:
        checkpoint = dict(state_dict=net.state_dict())
        checkpoint_path = os.path.join("code/checkpoints", config.checkpoint_id + ".pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        reparameterize(config, net, device, True)


if __name__ == "__main__":
    args = train_parser()

    train(args)
