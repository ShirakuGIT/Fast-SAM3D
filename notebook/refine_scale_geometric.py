#!/usr/bin/env python3
"""
refine_scale_direct.py

Automatically computes the metric scale correction for a SAM‑3D rough metric mesh
by comparing the mesh's projected Z‑values with the real depth image inside the object mask.

Usage:
    python refine_scale_direct.py \
        --data_dir ../SceneComplete/.../grasp_data \
        --mesh_dir output_scene_final/objects \
        --obj_id 0 \
        --output_dir output_scene_final/objects/scaled
"""

import os, sys, json, argparse, numpy as np
import cv2
import trimesh

def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    K = np.array(data['intrinsic_matrix']).reshape(3,3)
    return K

def estimate_scale_direct(mesh, depth_metric, mask, K):
    """
    Compute the scale factor (real/metric) by comparing mesh vertex depths
    with the real depth map inside the mask.

    mesh: trimesh.Trimesh (vertices in camera frame, metric)
    depth_metric: 2D numpy array (H, W) in metres
    mask: 2D numpy array (H, W) uint8 (0 or 255)
    K: camera intrinsics (3x3)
    Returns: float scale factor
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (N,3) in camera coords
    # Project to pixel coordinates
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X, Y, Z = verts[:,0], verts[:,1], verts[:,2]
    # Pinhole projection
    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    # Round to nearest pixel
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)
    # Keep only points that project inside the image and inside the mask
    H, W = depth_metric.shape
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    u_val = u_int[valid]
    v_val = v_int[valid]
    Z_mesh = Z[valid]
    # Mask check
    mask_valid = mask[v_val, u_val] > 128
    u_val = u_val[mask_valid]
    v_val = v_val[mask_valid]
    Z_mesh = Z_mesh[mask_valid]
    # Get real depth at those pixels
    real_depth = depth_metric[v_val, u_val]
    # Filter out zeros/invalid
    good = real_depth > 0
    if np.sum(good) < 10:
        raise RuntimeError("Too few overlapping points between mesh and mask. Check rough metric mesh.")
    Z_mesh = Z_mesh[good]
    real_depth = real_depth[good]
    # Scale factor: each point gives real_depth / Z_mesh
    ratios = real_depth / Z_mesh
    # Use median to be robust to outliers
    scale = np.median(ratios)
    return scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--obj_id", type=int, default=0)
    parser.add_argument("--output_dir", default="./scaled")
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = args.mesh_dir
    obj_id = args.obj_id
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    mask_path = os.path.join(data_dir, f"{obj_id}_mask.png")
    depth_path = os.path.join(data_dir, f"{obj_id}_depth.png")
    intrinsics_path = os.path.join(data_dir, "cam_K.json")
    mesh_path = os.path.join(mesh_dir, f"object_{obj_id}_rough_metric.glb")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    depth_metric = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * 0.001  # mm -> m
    K = load_intrinsics(intrinsics_path)

    print(f"Loading rough metric mesh: {mesh_path}")
    scene_or_mesh = trimesh.load(mesh_path)
    # If scene, merge all geometry
    if isinstance(scene_or_mesh, trimesh.Scene):
        verts_list, faces_list = [], []
        offset = 0
        for g in scene_or_mesh.geometry.values():
            v = np.asarray(g.vertices)
            f = np.asarray(g.faces) + offset
            verts_list.append(v)
            faces_list.append(f)
            offset += len(v)
        if not verts_list:
            raise ValueError("Empty scene")
        mesh = trimesh.Trimesh(vertices=np.concatenate(verts_list),
                               faces=np.concatenate(faces_list))
    else:
        mesh = scene_or_mesh

    # Compute scale
    scale = estimate_scale_direct(mesh, depth_metric, mask, K)
    print(f"Estimated scale factor: {scale:.4f}")

    # Apply scale and export
    scaled_verts = np.asarray(mesh.vertices) * scale
    mesh.vertices = scaled_verts
    out_path = os.path.join(args.output_dir, f"object_{obj_id}_scaled.glb")
    # Export as scene to preserve material info
    if isinstance(scene_or_mesh, trimesh.Scene):
        for g in scene_or_mesh.geometry.values():
            g.vertices *= scale
        scene_or_mesh.export(out_path)
    else:
        mesh.export(out_path)
    print(f"Scaled mesh saved to {out_path}")

if __name__ == "__main__":
    main()