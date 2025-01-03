import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, modified_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    print(dataset.eval)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        test_cameara = scene.getTestCameras()

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("-m", default="Grammar",action="store_true")
    
    
    # Change back to the origninal version 

    
    
    
    # directory_path = '/data/guest_storage/zhanpengluo/copy_gs/gaussian-splatting/output/SGD_Evaluation/tant'
    # file_paths = [os.path.join(directory_path, name) for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    file_paths = ["/data/guest_storage/zhanpengluo/original_implementation/gaussian-splatting/output/Adam_F2Nerf/stair"]

    # args.model_path = 'NOT EXIST'

    for path in file_paths:
        args = modified_args(parser,path)
        model_para =  model.extract(args)
        model_para.model_path  = path
        print("Rendering " + model_para.model_path)

        # Initialize system state (RNG)
        safe_state(args.quiet)

        render_sets(model_para, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
        
# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train",default= True, action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)