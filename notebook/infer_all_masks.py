import sys, os, time, glob
import numpy as np
import torch
import argparse
from PIL import Image
from plyfile import PlyData, PlyElement
from omegaconf import OmegaConf
from inference import Inference, load_image

# Patch for your environment
os.environ['TORCH_HOME'] = 'checkpoints/torch-cache'

def load_mask_from_path(mask_path):
    from inference import load_image   # reuse the pipeline's own image loader
    mask = load_image(mask_path)       # returns tensor [1, H, W] or [H, W]
    mask = mask > 0                    # Boolean tensor
    if mask.ndim == 3:
        mask = mask[..., -1]           # if RGBA, take alpha channel
    return mask

def combine_ply_files(ply_paths, output_path):
    """Merge multiple Gaussian splat PLY files into one."""
    all_vertices = []
    for p in ply_paths:
        plydata = PlyData.read(p)
        verts = plydata['vertex'].data
        all_vertices.append(verts)

    combined = np.concatenate(all_vertices)
    el = PlyElement.describe(combined, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Combined PLY saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_image", type=str,
                        default="input_data/scene.png",
                        help="Full scene image used for segmentation")
    parser.add_argument("--mask_dir", type=str,
                        default="sam3_segmentation_results",
                        help="Folder containing mask_*.png files")
    parser.add_argument("--output_dir", type=str,
                        default="./output_scene",
                        help="Where to save individual and combined outputs")
    parser.add_argument("--tag", default="hf")
    parser.add_argument("--enable_acceleration", action="store_true",
                        help="Enable faster inference (recommended)")
    # Keep other hyperparams if needed
    args = parser.parse_args()

    # Load config once
    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)

    # Enable acceleration options
    if args.enable_acceleration:
        config['ss_generator_config_path'] = "ss_generator_faster.yaml"
        config['slat_generator_config_path'] = "slat_generator_faster.yaml"
        print("✅ Acceleration enabled")

    # Build inference pipeline (models loaded once)
    print("Loading models...")
    inference = Inference(config, compile=False, args=args)

    # Set parameters required by the pipeline
    from argparse import Namespace
    params_args = Namespace(
        ss_cache_stride=3,
        ss_warmup=2,
        ss_order=1,
        ss_momentum_beta=0.5,
        slat_thresh=0.5,
        slat_warmup=2,
        slat_carving_ratio=0.15,
        mesh_spectral_threshold_low=0.5,
        mesh_spectral_threshold_high=0.7,
        enable_mesh_aggregation=True,
    )
    inference.get_params(params_args)

    # Load the full scene image
    scene_image = load_image(args.scene_image)

    # Find all mask files
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "mask_*.png")))
    if not mask_files:
        print(f"No mask files found in {args.mask_dir}")
        return

    print(f"Found {len(mask_files)} masks. Processing...")

    individual_plys = []

    for mask_path in mask_files:
        # Extract object name from filename: "mask_<name>_0.png" -> "<name>"
        basename = os.path.basename(mask_path)
        parts = basename.replace("mask_", "").rsplit("_", 1)
        obj_name = parts[0] if len(parts)==2 else basename

        print(f"\n--- Processing {obj_name} ---")

        # Load mask
        mask = load_mask_from_path(mask_path)

        # Run inference
        t0 = time.time()
        output = inference(scene_image, mask, seed=42)
        dt = time.time() - t0
        print(f"Done in {dt:.1f}s")

        # Save individual results
        obj_dir = os.path.join(args.output_dir, obj_name)
        os.makedirs(obj_dir, exist_ok=True)

        ply_individual = os.path.join(obj_dir, f"{obj_name}.ply")
        glb_individual = os.path.join(obj_dir, f"{obj_name}.glb")

        # Save PLY (Gaussian splat)
        from plyfile import PlyData, PlyElement
        gs = output["gs"]
        xyz = gs._xyz.detach().cpu().numpy()
        f_dc = gs._features_dc.detach().cpu().numpy()
        SH_C0 = 0.28209479177387814
        rgb = np.clip(0.5 + SH_C0 * f_dc, 0, 1) * 255
        rgb = rgb.astype(np.uint8).squeeze(1)
        verts = np.empty(xyz.shape[0],
                         dtype=[('x','f4'),('y','f4'),('z','f4'),
                                ('red','u1'),('green','u1'),('blue','u1')])
        verts['x'], verts['y'], verts['z'] = xyz[:,0], xyz[:,1], xyz[:,2]
        verts['red'], verts['green'], verts['blue'] = rgb[:,0], rgb[:,1], rgb[:,2]
        PlyData([PlyElement.describe(verts, 'vertex')]).write(ply_individual)

        # Save GLB
        output["glb"].export(glb_individual)

        individual_plys.append(ply_individual)
        print(f"Saved: {ply_individual}")

    # Combine all individual PLYs into one scene
    if individual_plys:
        combined_ply = os.path.join(args.output_dir, "combined_scene.ply")
        combine_ply_files(individual_plys, combined_ply)
        print("\n🎉 All done. Combined scene saved as:", combined_ply)

if __name__ == "__main__":
    main()