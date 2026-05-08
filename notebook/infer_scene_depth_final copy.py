import sys, os, time, torch, numpy as np, argparse, imageio, glob
from omegaconf import OmegaConf
from inference import Inference, ready_gaussian_for_video_rendering, load_image, make_scene, render_video

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
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]; elements['y'] = xyz[:, 1]; elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]; elements['green'] = rgb[:, 1]; elements['blue'] = rgb[:, 2]
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    print(f"Saved colored PLY to {path}")

def main():
    parser = argparse.ArgumentParser(description="3D Scene Inference Script")
    parser.add_argument("--tag", type=str, default="hf")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./Generate/Scene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss_cache_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=int, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_carving_ratio", type=float, default=0.15)
    parser.add_argument("--mesh_spectral_threshold_low", type=float, default=0.5)
    parser.add_argument("--mesh_spectral_threshold_high", type=float, default=0.7)
    parser.add_argument("--enable_ss_cache", action="store_true")
    parser.add_argument("--enable_slat_carving", action="store_true")
    parser.add_argument("--enable_mesh_aggregation", action="store_true")
    parser.add_argument("--enable_acceleration", action="store_true")
    args, _ = parser.parse_known_args()

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

    # Bypass adaptive factor (still not needed)
    import sam3d_objects.pipeline.inference_pipeline as ip
    ip.calculate_adaptive_factor = lambda *args, **kwargs: (1.0, 0.0)

    image_name = os.path.basename(args.image_dir)
    image_path = os.path.join(args.image_dir, "scene_full_image.png")
    print(f"📂 Loading data from: {args.image_dir}")
    image = load_image(image_path)

    # Masks
    mask_files = sorted(glob.glob(os.path.join(args.image_dir, "*_mask.png")))
    mask_files = [f for f in mask_files if "_masked" not in os.path.basename(f)]
    if len(mask_files) == 0:
        raise FileNotFoundError("No *_mask.png files found")
    masks = []
    for f in mask_files:
        m = imageio.imread(f)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append((m > 128).astype(np.float32))
    print(f"Loaded {len(masks)} masks.")

    # Per‑object pointmaps
    hfer_files = sorted(glob.glob(os.path.join(args.image_dir, "*_hfer.npy")))
    hfer_files = [f for f in hfer_files if not os.path.basename(f).startswith("scene_")]
    if len(hfer_files) != len(masks):
        raise RuntimeError(f"Mismatch: {len(masks)} masks but {len(hfer_files)} pointmaps")
    hfers = [np.load(f) for f in hfer_files]
    print(f"Loaded {len(hfers)} per‑object pointmaps.")

    print(f"🚀 Begin Inference, total {len(masks)} objects...")

    outputs = []
    s_time = time.time()
    for i in range(len(masks)):
        torch.cuda.empty_cache()
        if hasattr(inference, 'get_hfer'):
            inference.get_hfer(hfers[i])
        if hasattr(inference, 'get_params'):
            inference.get_params(args)

        print(f"  -> Processing object {i+1}/{len(masks)}...")
        output = inference(image, masks[i], seed=args.seed)

        obj_dir = os.path.join(args.output_dir, "objects")
        os.makedirs(obj_dir, exist_ok=True)

        # Save canonical PLY and GLB
        save_visual_ply(output["gs"], os.path.join(obj_dir, f"object_{i}.ply"))
        output["glb"].export(os.path.join(obj_dir, f"object_{i}_canonical.glb"))
        print(f"   💾 Saved canonical PLY and GLB")

        # Save pose parameters
        pose_data = {
            "scale": output["scale"].cpu().numpy(),       # shape (1,3)
            "rotation": output["rotation"].cpu().numpy(), # quaternion (1,4)
            "translation": output["translation"].cpu().numpy()  # (1,3)
        }
        np.savez(os.path.join(obj_dir, f"object_{i}_pose.npz"), **pose_data)
        print(f"   💾 Saved pose parameters to object_{i}_pose.npz")

        # Apply SAM3D pose to get rough metric mesh (for visualization / initial)
        verts_canon = np.asarray(output["glb"].vertices, dtype=np.float32)
        scale = pose_data["scale"].flatten()
        quat = pose_data["rotation"].flatten()
        trans = pose_data["translation"].flatten()
        from pytorch3d.transforms import quaternion_to_matrix as quat2mat
        R = quat2mat(torch.tensor(quat)).numpy().reshape(3,3)
        metric_verts = (verts_canon * scale) @ R.T + trans
        rough_metric_mesh = output["glb"].copy()
        rough_metric_mesh.vertices = metric_verts
        rough_metric_mesh.export(os.path.join(obj_dir, f"object_{i}_rough_metric.glb"))
        print(f"   💾 Saved rough metric mesh (pre-DINO scaling)")

        outputs.append(output)

    e_time = time.time()
    print(f"⏱️ Total Inference Time: {e_time - s_time:.2f}s")

    # ... rest of scene composition unchanged ...
    print("🧩 Compositing scene...")
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)
    os.makedirs(args.output_dir, exist_ok=True)
    scene_ply = os.path.join(args.output_dir, f"{image_name}_scene.ply")
    save_visual_ply(scene_gs, scene_ply)
    print(f"Combined scene PLY saved: {scene_ply}")

    try:
        print("🎥 Rendering video...")
        video_frames = render_video(scene_gs, r=2.5, fov=60, resolution=1024)["color"]
        gif_path = os.path.join(args.output_dir, f"{image_name}.gif")
        imageio.mimsave(gif_path, video_frames, format="GIF", duration=1000/30, loop=0)
        print(f"GIF saved to: {gif_path}")
    except Exception as e:
        print(f"Video rendering failed (non‑critical): {e}")

if __name__ == "__main__":
    main()