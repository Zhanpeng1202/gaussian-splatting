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

 
def resize_feature_maps(feature_map, target_size):
    return F.interpolate(feature_map, size=target_size, mode='bilinear', align_corners=False)

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'vgg', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor, w:torch.TensorType, file_number:int):
        with torch.no_grad():
            feats1 = self.net.forward(x)
            feats2 = self.net.forward(y)
            feats3 = self.net.forward(w)
        
            target_size = feats1[0].shape[2:] 
            sgd_maps = []
            adam_maps = []
            diff_maps = []
            
            for f1, f2,f3 in zip(feats1, feats2,feats3):
                f1_resized = resize_feature_maps(f1, target_size).cpu()
                f2_resized = resize_feature_maps(f2, target_size).cpu()
                f3_resized = resize_feature_maps(f3, target_size).cpu()


                _sgd = torch.square(f1_resized - f2_resized).mean(1, keepdim=True).squeeze().cpu().numpy()
                _adam = torch.square(f3_resized - f2_resized).mean(1, keepdim=True).squeeze().cpu().numpy()
                _diff = _sgd - _adam
                
                sgd_maps.append(_sgd)
                adam_maps.append(_adam)
                diff_maps.append(_diff)

            
        sgd_em = np.mean(np.stack(sgd_maps), axis=0)
        adam_em = np.mean(np.stack(adam_maps),axis=0)
        diff_em = np.mean(np.stack(diff_maps), axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        
        axes[0][0].imshow(sgd_em, cmap='jet')
        axes[0][0].set_title('SGD_LPIPS')
        axes[0][0].axis('off')  
        
        axes[0][1].imshow(adam_em, cmap='jet')
        axes[0][1].set_title('Adam_LPIPS')
        axes[0][1].axis('off')
        
        axes[1][0].imshow(diff_em, cmap='jet')
        axes[1][0].set_title('LPIPS_difference')
        axes[1][0].axis('off')   
        
        axes[1][1].imshow(y.squeeze(0).permute(1,2,0).cpu())
        axes[1][1].set_title('gt')
        axes[1][1].axis('off')   
        
        
        dir_path = "/data/guest_storage/zhanpengluo/copy_gs/gaussian-splatting/lpips_visualize/counter/"
        file_name = dir_path+str(file_number).zfill(5) +".png"

        plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
        plt.close()


        return 

def lpips(x: torch.Tensor,
          y: torch.Tensor,
          w:torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1',
          file_number:int = 0):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y,w,file_number)

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
        lpips(sgd_renders[idx], sgd_gts[idx],
              adam_renders[idx],
              net_type='vgg',file_number=file_number)
        file_number+=1






if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', default="/data/guest_storage/zhanpengluo/copy_gs/gaussian-splatting/output/counter_new_param", required=False, nargs="+", type=str)
    args = parser.parse_args()
    adam_path = "/data/guest_storage/zhanpengluo/gaussian-splatting/Adam_output/GS_Adam_counter"
    evaluate(args.model_paths,adam_path)
