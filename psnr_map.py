#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import torch.nn as nn

from lpipsPyTorch.modules.networks import get_network, LinLayers
from lpipsPyTorch.modules.utils import get_state_dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

counter = 0

def normalize_error_map(error_map, eps=1e-8):
    """Normalize the error map for better visualization."""
    min_val = error_map.min()
    max_val = error_map.max()
    return (error_map - min_val) / (max_val - min_val + eps)

def PSNR_error_map(sgd,adam,gt):
    
    # sgd_em = (torch.abs(sgd - gt)).mean(dim=0).cpu().permute(1,2,0)*100
    # adam_em = (torch.abs(adam - gt)).mean(dim=0).cpu().permute(1,2,0)*100
    # diff_em = (torch.abs(sgd - adam)).mean(dim=0).cpu().permute(1,2,0)*100
    
    sgd_em = ((sgd - gt)**2).mean(dim=0).cpu().permute(1,2,0)*100
    adam_em = ((adam - gt)**2).mean(dim=0).cpu().permute(1,2,0)*100
    diff_em = ((sgd - adam)**2).mean(dim=0).cpu().permute(1,2,0)*100
    
    
    # sgd_em = normalize_error_map(sgd_em)
    # adam_em = normalize_error_map(adam_em)
    # diff_em = normalize_error_map(diff_em)

    # Plot the maps
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    
    # axes[0][0].imshow(sgd_em, cmap='jet')
    # axes[0][0].set_title('SGD_PSNR')
    # axes[0][0].axis('off')  
    
    # axes[0][1].imshow(adam_em, cmap='jet')
    # axes[0][1].set_title('Adam_PSNR')
    # axes[0][1].axis('off')
    
    axes[0][0].imshow(sgd.squeeze(0).permute(1,2,0).cpu())
    axes[0][0].set_title('SGD_PSNR')
    axes[0][0].axis('off')  
    
    axes[0][1].imshow(adam.squeeze(0).permute(1,2,0).cpu())
    axes[0][1].set_title('Adam_PSNR')
    axes[0][1].axis('off')
    
    axes[1][0].imshow(diff_em, cmap='jet')
    axes[1][0].set_title('PSNR_difference')
    axes[1][0].axis('off')   
    
    axes[1][1].imshow(gt.squeeze(0).permute(1,2,0).cpu())
    axes[1][1].set_title('gt')
    axes[1][1].axis('off')   
    
    global counter
    dir_path = "/data/guest_storage/zhanpengluo/visualization/psnr/f2nerf/stair_compare/"
    file_name = dir_path+str(counter).zfill(5) +".png"
    counter+=1

    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(sgd_path,adam_path):
    
    sgd_dir = Path(sgd_path) / "test"
    adam_dir = Path(adam_path) / "test"
    file_number = 0
    method =  os.listdir(sgd_dir)[0]
    
    sgd_method_dir = sgd_dir / method
    sgd_gt_dir = sgd_method_dir/ "gt"
    sgd_renders_dir = sgd_method_dir / "renders"
    
    adam_method_dir = adam_dir / method
    adam_gt_dir = adam_method_dir/ "gt"
    adam_renders_dir = adam_method_dir / "renders"
    
    
    sgd_renders, sgd_gts, image_names = readImages(sgd_renders_dir, sgd_gt_dir)
    adam_renders, adam_gts, image_names = readImages(adam_renders_dir, adam_gt_dir)
    
    
    for idx in tqdm(range(len(sgd_renders)), desc="Metric evaluation progress"):
        PSNR_error_map(sgd_renders[idx], sgd_gts[idx],adam_renders[idx])


if __name__ == "__main__":
    device = torch.device("cuda:7")
    torch.cuda.set_device(device)
    counter = 0

    adam_path  = "/data/guest_storage/zhanpengluo/original_implementation/gaussian-splatting/output/Adam_F2Nerf/stair"
    sgd_path = "/data/guest_storage/zhanpengluo/newcopy_gs/gaussian-splatting/output/test/stair"
    evaluate(sgd_path,adam_path)
