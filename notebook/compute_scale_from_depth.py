#!/usr/bin/env python3
"""
compute_scale_from_depth.py

Automatic scale correction: compares mesh vertex depths against real depth
inside the object mask. No DINO, no manual factors. Uses the per‑object
pointmap (which is already in metres) as ground truth.

Usage:
    python compute_scale_from_depth.py \
        --data_dir ../SceneComplete/.../grasp_data \
        --mesh_dir output_scene_final/objects \
        --obj_id 0
"""

import os, sys, json, argparse, numpy as np
import cv2
import trimesh

def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data['intrinsic_matrix']).reshape(3, 3)

def compute_scale_from_depth(verts, hfer, mask, K):
    """
    verts: (N,3) mesh vertices in camera frame (metres)
    hfer: (H,W,3) per‑object pointmap in metres
    mask: (H,W) uint8 object mask (255 = object)
    K: (3,3) intrinsics
    Returns: scale factor (float)
    """
    H, W = hfer.shape[:2]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    X, Y, Z = verts[:,0], verts[:,1], verts[:,2]

    # Project to pixels
    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    # Keep only points that project inside the image AND inside the mask
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (Z > 0)
    ui, vi, Z_mesh = ui[inside], vi[inside], Z[inside]

    mask_vals = mask[vi, ui]
    in_mask = mask_vals > 128
    ui, vi, Z_mesh = ui[in_mask], vi[in_mask], Z_mesh[in_mask]

    if len(Z_mesh) < 10:
        raise RuntimeError(f"Only {len(Z_mesh)} mesh vertices project inside mask")

    # Get real depth from pointmap at those pixels
    real_Z = hfer[vi, ui, 2]  # Z channel of pointmap

    # Filter out zeros (invalid depth)
    valid = real_Z > 0
    real_Z, Z_mesh = real_Z[valid], Z_mesh[valid]

    if len(real_Z) < 10:
        raise RuntimeError("Not enough valid depth pixels inside mask")

    # Scale factor: median of real_Z / mesh_Z
    ratios = real_Z / Z_mesh
    scale = np.median(ratios)

    print(f"  Valid correspondences: {len(real_Z)}")
    print(f"  Median ratio (real_Z / mesh_Z): {scale:.4f}")
    return scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--obj_id", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = args.mesh_dir
    obj_id = args.obj_id

    # Load inputs
    mask_path = os.path.join(data_dir, f"{obj_id}_mask.png")
    hfer_path = os.path.join(data_dir, f"{obj_id}_hfer.npy")
    intrinsics_path = os.path.join(data_dir, "cam_K.json")
    mesh_path = os.path.join(mesh_dir, f"object_{obj_id}_rough_metric.glb")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    hfer = np.load(hfer_path)           # (H, W, 3) in metres
    K = load_intrinsics(intrinsics_path)

    print(f"Loading mesh: {mesh_path}")
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        verts = np.concatenate([g.vertices for g in scene.geometry.values()], axis=0)
    else:
        verts = np.asarray(scene.vertices)
    print(f"Mesh has {len(verts)} vertices")

    scale = compute_scale_from_depth(verts, hfer, mask, K)
    print(f"\nCorrected scale factor: {scale:.6f}")
    print(f"Multiply rough metric mesh by this factor to get true metric scale.")

    # Save factor
    with open("scale_correction.txt", "w") as f:
        f.write(f"{scale}\n")
    print("Saved scale_correction.txt")

if __name__ == "__main__":
    main()