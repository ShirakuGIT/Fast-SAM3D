import sys
import os
import time
import torch
import numpy as np
import argparse
import imageio
from omegaconf import OmegaConf
from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, load_hfers, make_scene, render_video

sys.path.append("notebook")
os.environ['TORCH_HOME'] = 'checkpoints/torch-cache'

def save_visual_ply(gs_model, path):
    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = gs_model._xyz.detach().cpu().numpy()
    f_dc = gs_model._features_dc.detach().contiguous().cpu().numpy()
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + (SH_C0 * f_dc)
    rgb = np.clip(rgb, 0, 1) * 255
    rgb = rgb.astype(np.uint8).squeeze(1)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    print(f"Saved colored PLY to {path}")

def main():
    parser = argparse.ArgumentParser(description="3D Scene Inference Script")
    parser.add_argument("--tag", type=str, default="hf")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./Generate/Scene")
    parser.add_argument("--seed", type=int, default=42)
    # SSG params
    parser.add_argument("--ss_cache_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=int, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)
    # SLaT params
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_carving_ratio", type=float, default=0.15)
    # Mesh params
    parser.add_argument("--mesh_spectral_threshold_low", type=float, default=0.5)
    parser.add_argument("--mesh_spectral_threshold_high", type=float, default=0.7)
    # Flags
    parser.add_argument("--enable_ss_cache", action="store_true")
    parser.add_argument("--enable_slat_carving", action="store_true")
    parser.add_argument("--enable_mesh_aggregation", action="store_true")
    parser.add_argument("--enable_acceleration", action="store_true")
    args, _ = parser.parse_known_args()

    # Enable all acceleration flags if --enable_acceleration is set
    if args.enable_acceleration:
        args.enable_ss_cache = True
        args.enable_slat_carving = True
        args.enable_mesh_aggregation = True
    print(f"✅ SS:{args.enable_ss_cache}, SLaT:{args.enable_slat_carving}, Mesh:{args.enable_mesh_aggregation}")

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    if args.enable_ss_cache:
        config['ss_generator_config_path'] = "ss_generator_faster.yaml"
    if args.enable_slat_carving:
        config['slat_generator_config_path'] = "slat_generator_faster.yaml"

    inference = Inference(config, compile=False, args=args)

    image_name = os.path.basename(args.image_dir)
    image_path = os.path.join(args.image_dir, "image.png")
    print(f"📂 Loading data from: {args.image_dir}")
    image = load_image(image_path)
    masks = load_masks(args.image_dir, extension=".png")
    hfers = load_hfers(args.image_dir, extension=".png")
    print(f"🚀 Begin Inference, total {len(masks)} objects...")

    outputs = []   # collect per-object outputs for final composition
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

        outputs.append(output)

    e_time = time.time()
    print(f"⏱️ Total Inference Time: {e_time - s_time:.2f}s")

    # ---- Compose final scene using the official function ----
    print("🧩 Compositing scene...")
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    # ---- Save combined PLY & render video ----
    os.makedirs(args.output_dir, exist_ok=True)
    scene_ply = os.path.join(args.output_dir, f"{image_name}_scene.ply")
    save_visual_ply(scene_gs, scene_ply)
    print(f"Combined scene PLY saved: {scene_ply}")

    print("🎥 Rendering video...")
    video_frames = render_video(
        scene_gs,
        r=2.5,
        fov=60,
        resolution=1024,
    )["color"]

    gif_path = os.path.join(args.output_dir, f"{image_name}.gif")
    imageio.mimsave(gif_path, video_frames, format="GIF", duration=1000/30, loop=0)
    print(f"GIF saved to: {gif_path}")

if __name__ == "__main__":
    main()