#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import sys
import torch

if __name__ == "__main__":
    input = 'checkpoints/pixpro_base_r50_400ep_md5_919c6612.pth'
    # input = '/home/hungld11/Intern/khanghn1/PixPro/output/pixpro_base_r50_100ep/ckpt_epoch_195.pth'

    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]
    # obj = obj['state_dict']
    # print(obj)

    new_model = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder.") or k.endswith('num_batches_tracked'):
            continue
        old_k = k
        k = k.replace("module.encoder.", "")
        print(k)
        new_model[k] = v
        
        # if "layer" not in k:
        #     k = "stem." + k
        # for t in [1, 2, 3, 4]:
        #     k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        # for t in [1, 2, 3]:
        #     k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        # k = k.replace("downsample.0", "shortcut")
        # k = k.replace("downsample.1", "shortcut.norm")
        # print(old_k, "->", k)
        # new_model[k] = v.numpy()

    res = {
        "state_dict": new_model,
        "__author__": "PixPro",
        "matching_heuristics": True}
    torch.save(res, 'checkpoints/converted_resnet50_pixpro_400ep_official_origin.pth')
