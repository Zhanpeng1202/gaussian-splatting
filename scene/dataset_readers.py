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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    
    if len(cam_centers) == 1:
        radius = 2
    else:
        radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # sys.stdout.write('\r')
        # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def createWallInitialization():
    cam_infos = []
    
    cam_name = "Virtual Camera"
    image_path = '/data/guest_storage/zhanpengluo/gaussian-splatting/GT.png'
    image = Image.open(image_path)
    R = np.eye(3)  
    T = np.zeros(3)
    
   
    

    FovY = 0.8753
    FovX = 1.4028

    cam_infos.append(CameraInfo(uid=0, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    image_path=image_path, image_name=cam_name, width=image.size[0], height=image.size[1] ))
    
    nerf_normalization = getNerfppNorm(cam_infos)
    
    step = 0.005
    x1 = np.arange(-0.1, 0, step)
    x2 = np.arange(0, 1, step*10)
    
    y1 = np.arange(-0.1, 0.1, step)
    y2 = np.arange(-1, 1, step*10)
    
    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    x1,x2 = x1.ravel(),x2.ravel()
    y1,y2 = y1.ravel(),y2.ravel()
    
    x = np.hstack([x1,x2])
    y = np.hstack([y1,y2])
    
    num_points1 = len(x1)
    num_points2 = len(x2)
    
    z1 = np.full(num_points1, 0.4)
    z2 = np.full(num_points2, 4)
    z = np.hstack([z1, z2])

    r1,r2 = np.zeros(num_points1),np.zeros(num_points2)
    g1,g2 = np.zeros(num_points1),np.zeros(num_points2)
    b1,b2 = np.zeros(num_points1),np.zeros(num_points2)

    nx_wall = np.full((num_points1+num_points2), 0.5)

    n_stripes = 5
    stripe_boundaries = np.linspace(-0.1, 0.1, n_stripes+1)
    for i in range(n_stripes):
        if i % 2 == 0:
            mask = (y1 >= stripe_boundaries[i]) & (y1 < stripe_boundaries[i+1])
            r1[mask] = 255  
            g1[mask] = 0    
            b1[mask] = 0    
        else:
            mask = (y1 >= stripe_boundaries[i]) & (y1 < stripe_boundaries[i+1])
            r1[mask] = 0    
            g1[mask] = 0    
            b1[mask] = 255  
            
    stripe_boundaries = np.linspace(-1, 1, n_stripes+1)
    for i in range(n_stripes):
        if i % 2 == 0:
            mask = (y2 >= stripe_boundaries[i]) & (y2 < stripe_boundaries[i+1])
            r2[mask] = 255  
            g2[mask] = 0    
            b2[mask] = 0    
        else:
            mask = (y2 >= stripe_boundaries[i]) & (y2 < stripe_boundaries[i+1])
            r2[mask] = 0    
            g2[mask] = 0    
            b2[mask] = 255  
    r = np.hstack([r1, r2])
    b = np.hstack([b1, b2])
    g = np.hstack([g1, g2])

    
    positions = np.vstack([x , y , z]).T
    colors = np.vstack([r, g, b]).T / 255.0
    normals = np.vstack([nx_wall, nx_wall, nx_wall]).T


    pcd =  BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=cam_infos,
                           test_cameras=cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=image_path)        
    return scene_info 

def randomPointInit():
    cam_infos = []
    
    cam_name = "Virtual Camera"
    image_path = '/data/guest_storage/zhanpengluo/gaussian-splatting/GT.png'
    image = Image.open(image_path)
    R = np.eye(3)  
    T = np.zeros(3)
    
   
    

    FovY = 0.8753
    FovX = 1.4028

    cam_infos.append(CameraInfo(uid=0, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    image_path=image_path, image_name=cam_name, width=image.size[0], height=image.size[1] ))
    
    nerf_normalization = getNerfppNorm(cam_infos)
    
    step = 0.0002
    x1 = np.arange(-0.1, 0, step)
    x2 = np.arange(0, 1, step*10)
    
    y1 = np.arange(-0.1, 0.1, step)
    y2 = np.arange(-1, 1, step*10)
    
    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    x1,x2 = x1.ravel(),x2.ravel()
    y1,y2 = y1.ravel(),y2.ravel()
    

    
    num_points1 = len(x1)
    num_points2 = len(x2)
    
    z1 = np.full(num_points1, 0.4)
    z2 = np.full(num_points2, 4)
    z = np.hstack([z1, z2])

    r1,r2 = np.zeros(num_points1),np.zeros(num_points2)
    g1,g2 = np.zeros(num_points1),np.zeros(num_points2)
    b1,b2 = np.zeros(num_points1),np.zeros(num_points2)

    nx_wall = np.full((num_points1+num_points2), 0.5)

    n_stripes = 5
    stripe_boundaries = np.linspace(-0.1, 0.1, n_stripes+1)
    for i in range(n_stripes):
        if i % 2 == 0:
            mask = (y1 >= stripe_boundaries[i]) & (y1 < stripe_boundaries[i+1])
            r1[mask] = 255  
            g1[mask] = 0    
            b1[mask] = 0    
        else:
            mask = (y1 >= stripe_boundaries[i]) & (y1 < stripe_boundaries[i+1])
            r1[mask] = 0    
            g1[mask] = 0    
            b1[mask] = 255  
            
    stripe_boundaries = np.linspace(-1, 1, n_stripes+1)
    for i in range(n_stripes):
        if i % 2 == 0:
            mask = (y2 >= stripe_boundaries[i]) & (y2 < stripe_boundaries[i+1])
            r2[mask] = 255  
            g2[mask] = 0    
            b2[mask] = 0    
        else:
            mask = (y2 >= stripe_boundaries[i]) & (y2 < stripe_boundaries[i+1])
            r2[mask] = 0    
            g2[mask] = 0    
            b2[mask] = 255  
    r = np.hstack([r1, r2])
    b = np.hstack([b1, b2])
    g = np.hstack([g1, g2])

    # small jitter
    #  
    jitter_bound= 0.005
    jitter_near_x = np.random.uniform(low=-jitter_bound, high=jitter_bound, size=len(x1))
    jitter_far_x  = np.random.uniform(low=-jitter_bound*10, high=jitter_bound*10, size=len(x2))
    x1 = x1+jitter_near_x
    x2 = x2+jitter_far_x
    
    jitter_near_y = np.random.uniform(low=-jitter_bound, high=jitter_bound, size=len(y1))
    jitter_far_y  = np.random.uniform(low=-jitter_bound*10, high=jitter_bound*10, size=len(y2))
    y1 = y1+jitter_near_y
    y2 = y2+jitter_far_y
    
    x = np.hstack([x1,x2])
    y = np.hstack([y1,y2])
    
    positions = np.vstack([x , y , z]).T
    colors = np.vstack([r, g, b]).T / 255.0
    normals = np.vstack([nx_wall, nx_wall, nx_wall]).T


    pcd =  BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=cam_infos,
                           test_cameras=cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=image_path)        
    return scene_info 
 

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Wall_Expriment":createWallInitialization,
    "jitter_wall":randomPointInit
    # "scale_wall":scale
}