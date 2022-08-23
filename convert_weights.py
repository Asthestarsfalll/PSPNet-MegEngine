import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn
from models.pspnet import PSPNet
from models.torch_model import PSPNet as torch_PSPNet
model_name = "resnet50"

LAYER_MAPPER = {
    "resnet50": 50,
    "resnet101": 101,
    "resnet152": 152,
}

def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert(torch_model, torch_dict):
    new_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        new_dict[k] = data
    return new_dict


def main(torch_name, torch_path,  num_classes):
    torch_state_dict = torch.load(torch_path, map_location='cpu')['state_dict']
    t = {}
    for k in torch_state_dict.keys():
        n_k = k.replace('module.', '')
        t[n_k] = torch_state_dict[k]
    torch_model = torch_PSPNet(layers=LAYER_MAPPER[model_name], classes=num_classes)
    torch_model.load_state_dict(t)

    new_dict = convert(torch_model, t)
    model = PSPNet(layers=LAYER_MAPPER[model_name], classes=num_classes)

    error = model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='resnet50',
        help=f"Path to torch saved model, default: {list(LAYER_MAPPER)}",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="train_epoch_200.pth",
        help=f"Path to torch saved model, default: None",
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        type=int,
        default=19,
    )
    args = parser.parse_args()
    main(args.model, args.dir, args.num_classes)
