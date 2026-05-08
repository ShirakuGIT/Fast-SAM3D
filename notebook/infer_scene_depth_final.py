import sys, os, time, torch, numpy as np, argparse, imageio, glob
from omegaconf import OmegaConf
from inference import Inference, ready_gaussian_for_video_rendering, load_image, make_scene, render_video
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

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
    parser = argparse.ArgumentParser(description="3D Scene Inference (community metric fix)")
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

    # Required by Fast‑SAM3D pipeline – set SS/SLaT/Mesh parameters
    inference.get_params(args)   # <--- add this line

    # Bypass adaptive factor (still needed)
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

    # Per‑object pointmaps (OpenCV frame)
    hfer_files = sorted(glob.glob(os.path.join(args.image_dir, "*_hfer.npy")))
    hfer_files = [f for f in hfer_files if not os.path.basename(f).startswith("scene_")]
    if len(hfer_files) != len(masks):
        raise RuntimeError(f"Mismatch: {len(masks)} masks but {len(hfer_files)} pointmaps")
    hfers = [np.load(f) for f in hfer_files]   # each (H,W,3) in OpenCV (X right, Y down, Z forward)

    print(f"🚀 Begin Inference, total {len(masks)} objects...")

    outputs = []
    s_time = time.time()

    for i in range(len(masks)):
        torch.cuda.empty_cache()

        # Convert pointmap from OpenCV to PyTorch3D frame (exactly as in the reference notebook)
        pt_opencv = hfers[i].astype(np.float32)                     # (H,W,3)
        pt_pt3d = np.stack([-pt_opencv[...,0], -pt_opencv[...,1], pt_opencv[...,2]], axis=-1)
        # Replace zeros with NaN (as in the reference)
        pt_pt3d[pt_pt3d == 0] = np.nan

        pointmap_tensor = torch.tensor(pt_pt3d, dtype=torch.float32)

        # Run inference with pointmap
        print(f"  -> Processing object {i+1}/{len(masks)}...")
        output = inference(image, masks[i], seed=args.seed, pointmap=pointmap_tensor)

        obj_dir = os.path.join(args.output_dir, "objects")
        os.makedirs(obj_dir, exist_ok=True)

        # Save canonical PLY and GLB
        save_visual_ply(output["gs"], os.path.join(obj_dir, f"object_{i}.ply"))
        output["glb"].export(os.path.join(obj_dir, f"object_{i}_canonical.glb"))
        print(f"   💾 Saved canonical PLY and GLB")

                # ===== Exact community transformation (NumPy version – robust to NaN) =====
        R_yup_to_zup = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
        R_flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        R_pytorch3d_to_cam = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

        # Extract pose as NumPy arrays (squeeze batch dim)
        S = output["scale"].cpu().numpy().flatten()          # (3,)
        quat = output["rotation"].cpu().numpy().flatten()    # (4,)
        T = output["translation"].cpu().numpy().flatten()    # (3,)

        # Quaternion to rotation matrix (using SciPy)
        from scipy.spatial.transform import Rotation
        R_mat = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()  # SciPy expects (x,y,z,w), PyTorch3D uses (w,x,y,z)
        # If quat is (w,x,y,z), then we should use as_quat? Actually quat from PyTorch3D is (w,x,y,z), SciPy expects (x,y,z,w). So we reorder.
        # Better to use pytorch3d's quaternion_to_matrix but keep everything numpy.
        from pytorch3d.transforms import quaternion_to_matrix as q2m
        R_mat = q2m(torch.tensor(quat)).numpy().reshape(3,3)

        # 1. Convert canonical vertices from Y‑up to Z‑up (with Z‑flip)
        verts_canon_np = np.asarray(output["glb"].vertices, dtype=np.float32)
        verts_zup = (verts_canon_np @ R_flip_z) @ R_yup_to_zup

        # 2. Apply pose: scale → rotate → translate
        metric_verts_zup = (verts_zup * S) @ R_mat.T + T

        # 3. Convert PyTorch3D camera → OpenCV camera
        metric_verts_cam = metric_verts_zup @ R_pytorch3d_to_cam

        # Export final metric mesh
        metric_mesh = output["glb"].copy()
        metric_mesh.vertices = metric_verts_cam
        metric_mesh.export(os.path.join(obj_dir, f"object_{i}_true_metric.glb"))
        print(f"   💾 Saved true metric mesh (community fix, NumPy)")

        outputs.append(output)

    e_time = time.time()
    print(f"⏱️ Total Inference Time: {e_time - s_time:.2f}s")

    # Scene composition and rendering (unchanged)
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