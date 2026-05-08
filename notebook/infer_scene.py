import sys
import os
import time
import shutil
import torch
import numpy as np
import argparse
import imageio
from omegaconf import OmegaConf
from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, load_hfers, make_scene, render_video
from pytorch3d.transforms import quaternion_multiply, quaternion_invert
from sam3d_objects.utils.visualization import SceneVisualizer
import copy
import sys, os, time, torch, numpy as np, argparse, imageio, copy

sys.path.append("notebook")
os.environ['TORCH_HOME'] = 'checkpoints/torch-cache'

def save_visual_ply(gs_model, path):
    from plyfile import PlyData, PlyElement
    folder_path = os.path.dirname(path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    xyz = gs_model._xyz.detach().cpu().numpy()
    f_dc = gs_model._features_dc.detach().contiguous().cpu().numpy()
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + (SH_C0 * f_dc)
    
    rgb = np.clip(rgb, 0, 1) * 255
    rgb = rgb.astype(np.uint8)
    rgb = rgb.squeeze(1)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved colored PLY to {path}")

def main():
    parser = argparse.ArgumentParser(description="3D Scene Inference Script")

    parser.add_argument("--tag", type=str, default="hf", help="model Tag")
    parser.add_argument("--image_dir", type=str, required=True, help="image path")
    parser.add_argument("--output_dir", type=str, default="./Generate/Scene", help="output dir")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    
    # --- SSG  ---
    parser.add_argument("--ss_cache_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=int, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)
    
    # --- SLaT ---
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_carving_ratio", type=float, default=0.15)
    
    # --- Mesh ---
    parser.add_argument("--mesh_spectral_threshold_low", type=float, default=0.5)
    parser.add_argument("--mesh_spectral_threshold_high", type=float, default=0.7)
    
    parser.add_argument("--enable_ss_cache", action="store_true")
    parser.add_argument("--enable_slat_carving", action="store_true")
    parser.add_argument("--enable_mesh_aggregation", action="store_true")
    parser.add_argument("--enable_acceleration", action="store_true")
    
    args, unknown = parser.parse_known_args()


    def get_enable_params(args):
        args_dict = vars(args)
        enable_params = {k: v for k, v in args_dict.items() if k.startswith("enable_")}
        
        if enable_params.get('enable_acceleration', False):
            enable_params['enable_ss_cache'] = True
            enable_params['enable_slat_carving'] = True
            enable_params['enable_mesh_aggregation'] = True
        
        for k, v in enable_params.items():
            setattr(args, k, v)
        
        return enable_params

    enable_params = get_enable_params(args)
    print(f"✅: SS:{enable_params['enable_ss_cache']}, SLaT:{enable_params['enable_slat_carving']}, Mesh:{enable_params['enable_mesh_aggregation']}")

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"


    config = OmegaConf.load(config_path) 
    config.workspace_dir = os.path.dirname(config_path)
    
    if enable_params['enable_ss_cache']:
        config['ss_generator_config_path'] = "ss_generator_faster.yaml" 
    if enable_params['enable_slat_carving']:
        config['slat_generator_config_path'] = "slat_generator_faster.yaml" 

    inference = Inference(config, compile=False, args=args)

    image_name = os.path.basename(args.image_dir)
    image_path = os.path.join(args.image_dir,"image.png")
    print(f"📂 Loading data from: {args.image_dir}")
    image = load_image(image_path)
    masks = load_masks(args.image_dir, extension=".png")
    hfers = load_hfers(args.image_dir, extension=".png")

    print(f"🚀 Begin Inference, total {len(masks)} views...")

    s_time = time.time()
    
    # Lists to hold CPU tensors for the final composition
    all_xyz = []
    all_features_dc = []
    all_scaling = []
    all_rotation = []
    all_opacity = []
    min_kernels = []
    GaussianClass = None          # will be set from the first object

    s_time = time.time()

    for i in range(len(masks)):
        torch.cuda.empty_cache()
        if hasattr(inference, 'get_hfer'):
            inference.get_hfer(hfers[i])
        if hasattr(inference, 'get_params'):
            inference.get_params(args)

        print(f"  -> Processing object {i+1}/{len(masks)}...")
        output = inference(image, masks[i], seed=args.seed)

        # ---- Save per‑object outputs ----
        obj_dir = os.path.join(args.output_dir, "objects")
        os.makedirs(obj_dir, exist_ok=True)
        save_visual_ply(output["gs"], os.path.join(obj_dir, f"object_{i}.ply"))
        output["glb"].export(os.path.join(obj_dir, f"object_{i}.glb"))
        print(f"   💾 Saved object_{i}.ply and object_{i}.glb")

        # ---- Transform the Gaussian for the scene (pose correction) ----
        gs_current = output["gaussian"][0]
        gs_transformed = copy.deepcopy(gs_current)

        PC = SceneVisualizer.object_pointcloud(
            points_local=gs_current.get_xyz.unsqueeze(0),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        gs_transformed.from_xyz(PC.points_list()[0])
        gs_transformed.from_rotation(
            quaternion_multiply(
                quaternion_invert(output["rotation"]),
                gs_transformed.get_rotation,
            )
        )
        scale = gs_transformed.get_scaling
        adjusted_scale = scale * output["scale"]
        gs_transformed.mininum_kernel_size *= output["scale"][0, 0].item()
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                gs_transformed.mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        gs_transformed.from_scaling(adjusted_scale)

        # ---- Extract CPU tensors ----
        all_xyz.append(gs_transformed._xyz.detach().cpu())
        all_features_dc.append(gs_transformed._features_dc.detach().cpu())
        all_scaling.append(gs_transformed._scaling.detach().cpu())
        all_rotation.append(gs_transformed._rotation.detach().cpu())
        all_opacity.append(gs_transformed._opacity.detach().cpu())
        min_kernels.append(gs_transformed.mininum_kernel_size)

        # Remember the Gaussian class for later reconstruction
        if GaussianClass is None:
            GaussianClass = type(gs_current)

        # Free GPU memory of this object
        del output, gs_current, gs_transformed
        torch.cuda.empty_cache()

    e_time = time.time()
    print(f"⏱️ Total Inference Time: {e_time - s_time:.2f}s")

    # ── Build the final composed scene from CPU tensors ──
    print("🧩 Building final scene Gaussian...")
    device = inference._pipeline.device

    # Concatenate all tensors on CPU, then move to GPU
    final_xyz = torch.cat(all_xyz, dim=0).to(device)
    final_features_dc = torch.cat(all_features_dc, dim=0).to(device)
    final_scaling = torch.cat(all_scaling, dim=0).to(device)
    final_rotation = torch.cat(all_rotation, dim=0).to(device)
    final_opacity = torch.cat(all_opacity, dim=0).to(device)
    final_min_kernel = min(min_kernels)

    # Create the final Gaussian model (same class as the per-object ones)
    # Assumes the constructor accepts these keyword arguments
    scene_gs = GaussianClass(
        xyz=final_xyz,
        features_dc=final_features_dc,
        scaling=final_scaling,
        rotation=final_rotation,
        opacity=final_opacity,
        mininum_kernel_size=final_min_kernel,
    )

    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    # ── Save combined PLY and render video ──
    os.makedirs(args.output_dir, exist_ok=True)
    img_name = os.path.basename(args.image_dir)
    scene_ply = os.path.join(args.output_dir, f"{img_name}_scene.ply")
    save_visual_ply(scene_gs, scene_ply)
    print(f"Combined scene PLY saved: {scene_ply}")

    print("🎥 Rendering video...")
    video_frames = render_video(
        scene_gs,
        r=2.5,
        fov=60,
        resolution=1024,
    )["color"]

    gif_path = os.path.join(args.output_dir, f"{img_name}.gif")
    imageio.mimsave(gif_path, video_frames, format="GIF", duration=1000/30, loop=0)
    print(f"GIF saved to: {gif_path}")

   

if __name__ == "__main__":
    main()