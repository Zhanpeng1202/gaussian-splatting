import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from PIL import Image
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_sigma_selection
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import network_gui_ws
import time
import numpy as np
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def eulerRotation(theata,phi,psi):
    yaw = np.array([
        [math.cos(theata), 0 , math.sin(theata)],
        [0,1,0],
        [-math.sin(theata), 0 , math.cos(theata)],
    ])
    pitch = np.array([
        [1,0,0],
        [0,math.cos(phi),-math.sin(phi)],
        [0,math.sin(phi),math.cos(phi)],
    ])
    roll = np.array([
        [math.cos(psi),-math.sin(psi),0],
        [math.sin(psi),math.cos(psi),0],
        [0,0,1],
    ])
    
    return yaw@pitch@roll.tolist()
    
def visualize(optimizer,name, path):
    import matplotlib.pyplot as plt

    x = list(range(1,len(optimizer.data[name]["value_max"])+1))

    this_category = optimizer.data[name]
    plt.rcParams['path.simplify'] = False
    
    
    plt.figure(1).clf()
    plt.figure(1)
    plt.plot(x, this_category["value_max"], label="Max Opacity")  
    plt.plot(x, this_category["value_min"], label="Min Opacity")  
    plt.plot(x, this_category["value_mean"], label="Mean Opacity") 

    plt.savefig(f"{path}/{name}/value.png")

    plt.figure(2).clf()
    plt.figure(2)
    plt.plot(x, this_category["gradient_max"], label="Max Opacity")  
    plt.plot(x, this_category["gradient_min"], label="Min Opacity")  
    plt.plot(x, this_category["gradient_mean"], label="Mean Opacity") 


    plt.savefig(f"{path}/{name}/gradient.png")
    
    plt.figure(3).clf()
    plt.figure(3)
    plt.plot(x, this_category["stepsize_max"], label="Max Opacity")  
    plt.plot(x, this_category["stepsize_min"], label="Min Opacity")  
    plt.plot(x, this_category["stepsize_mean"], label="Mean Opacity") 


    plt.savefig(f"{path}/{name}/step_size.png")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,sigma_list):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    network_gui_ws.init_camera(scene)
    
    
    for iteration in range(first_iter, opt.iterations + 1):        

        # network_gui_ws.render_for_websocket(gaussians, pipe, background)
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # sigma_list = [5e-1, 2e-3]
        render_pkg = render_sigma_selection(viewpoint_cam, gaussians, pipe, bg, sigma_list=sigma_list)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
            
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # print(iteration)
            # if iteration % 10 == 0:
            #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            #     progress_bar.update(10)
            # if iteration == opt.iterations:
            #     progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in testing_iterations):
                print(f"Iterations: {iteration} & Number of points:{gaussians._xyz.shape[0]}")
            #     scene.save(iteration)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if(gaussians._xyz.shape[0]<6500000):
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # gaussians.optimizer.step(viewpoint_cam.getViewMatrix())
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                gaussians.optimizer_opacity.step()
                gaussians.optimizer_opacity.zero_grad(set_to_none=True)


        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        # return psnr_test




import itertools

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6129)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,  30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    model_arg = lp.extract(args)
    opt_arg   = op.extract(args)
    pipe_arg  = pp.extract(args)
    
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Tuning on the Truck dataset
    #
    

    print("----------This is our SGD evaluation on the MipNerf Datatset")

    
    position_lr_init  = [3]
    position_lr_final = [0.01]
    # 3000
    # feature_dc_lr =     [0.005,0.01,0.05]
    feature_dc_lr =     [0.005]
    sigma_s_list =      [1e-1 ,5e-2,5e-1]
    sigma_d_list =      [2e-4, 1e-4,5e-5]
    feature_rest_lr =   [600]
    rotation_lr =       [0.001]

    
    # directory_path = '/data/guest_storage/zhanpengluo/Dataset/MipNerf'
    directory_path = '/data/guest_storage/zhanpengluo/Dataset/F2Nerf'
    file_paths = [os.path.join(directory_path, name) for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    file_paths= ['/data/guest_storage/zhanpengluo/Dataset/MipNerf/bicycle']
    # file_paths = ['/data/guest_storage/zhanpengluo/Dataset/DL3DV-10K/1K/path1']

        
    
    param_grid = list(itertools.product(position_lr_init,position_lr_final, sigma_s_list,sigma_d_list,file_paths ))
    
    for xyz_init, xyz_final, simga_s, sigma_d,f_path in param_grid:   
        opt_arg.position_lr_init = xyz_init
        opt_arg.position_lr_final = xyz_final
        # opt_arg.scaling_lr = feat_dc
        # opt_arg.feature_dc_lr = feat_dc
        # opt_arg.feature_rest_lr = feat_rest
        # opt_arg.rotation_lr =rotate
        sigma_list = [simga_s, sigma_d]
        model_arg.source_path= f_path
        model_arg.model_path = os.path.join("/data/guest_storage/zhanpengluo/3dgs/newcopy_gs/gaussian-splatting/output/SGD_Evaluation/F2Nerf",os.path.basename(f_path))
        print(f"optimizing {os.path.basename(f_path)}")
        print(f"--------------  sigma_s and sigma_d: {sigma_list}")
        # if(os.path.basename(f_path)=='bicycle' or os.path.basename(f_path)=='grass'):
        #     # opt_arg.densify_until_iter = 10_000
        #     opt_arg.densify_grad_threshold = 0.00015
        
        training(model_arg,
             opt_arg,
             pipe_arg,
             args.test_iterations,
             args.save_iterations,
             args.checkpoint_iterations,
             args.start_checkpoint,
             args.debug_from,sigma_list)
        
    print("\nTraining complete.")
    sys.exit(0)

