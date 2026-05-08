#!/usr/bin/env python3
"""
refine_scale_automatic.py

Fully automatic metric scale recovery by aligning mesh vertex depths
with the real depth map, automatically handling any coordinate convention.

Usage:
    python refine_scale_automatic.py \
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

def project_and_check(verts, K, H, W, flip_y=False):
    """
    Projects vertices onto image plane.
    verts: (N,3) in camera frame.
    If flip_y is True, treats Y as pointing UP (OpenGL) and flips to OpenCV's Y-down.
    Returns projected pixel coords (u_int, v_int) and Z.
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = verts[:,0]
    Y = verts[:,1]
    Z = verts[:,2]
    if flip_y:
        Y = -Y   # flip from OpenGL (up) to OpenCV (down)
    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)
    # keep only those inside image
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    return u_int[valid], v_int[valid], Z[valid]

def estimate_scale_direct(mesh, depth_metric, mask, K):
    """
    Tries both possible coordinate conventions and returns the scale factor
    using the one that produces more overlapping points with the mask.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    H, W = depth_metric.shape

    # Try standard (Y down) and flipped (Y up)
    best_scale = None
    best_overlap = 0
    best_info = None

    for flip_y in [False, True]:
        u_int, v_int, Z_mesh = project_and_check(verts, K, H, W, flip_y=flip_y)
        # Now filter by mask
        mask_vals = mask[v_int, u_int] > 128
        u_f = u_int[mask_vals]
        v_f = v_int[mask_vals]
        Z_f = Z_mesh[mask_vals]
        # Get real depth
        real_depth = depth_metric[v_f, u_f]
        good = real_depth > 0   # valid depth
        if np.sum(good) < 10:
            continue
        ratio = real_depth[good] / Z_f[good]
        scale = np.median(ratio)
        overlap = np.sum(good)
        print(f"  Flipped Y: {flip_y} -> overlap pixels: {overlap}, scale: {scale:.4f}")
        if overlap > best_overlap:
            best_overlap = overlap
            best_scale = scale
            best_info = (flip_y, overlap, scale)

    if best_scale is None:
        raise RuntimeError("Could not find enough overlap with any convention. Check mesh position and mask.")

    print(f"Using flip_y={best_info[0]} with {best_info[1]} points, scale = {best_scale:.4f}")
    return best_scale

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

    mask_path = os.path.join(data_dir, f"{obj_id}_mask.png")
    depth_path = os.path.join(data_dir, f"{obj_id}_depth.png")
    intrinsics_path = os.path.join(data_dir, "cam_K.json")
    mesh_path = os.path.join(mesh_dir, f"object_{obj_id}_rough_metric.glb")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    depth_metric = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * 0.001
    K = load_intrinsics(intrinsics_path)

    print(f"Loading rough metric mesh: {mesh_path}")
    scene = trimesh.load(mesh_path)
    # Merge if scene
    if isinstance(scene, trimesh.Scene):
        all_verts = []
        all_faces = []
        off = 0
        for g in scene.geometry.values():
            v = np.asarray(g.vertices)
            f = np.asarray(g.faces) + off
            all_verts.append(v)
            all_faces.append(f)
            off += len(v)
        mesh = trimesh.Trimesh(vertices=np.concatenate(all_verts),
                               faces=np.concatenate(all_faces))
    else:
        mesh = scene

    scale = estimate_scale_direct(mesh, depth_metric, mask, K)
    print(f"Final scale factor: {scale:.4f}")

    # Apply scale
    mesh.vertices *= scale
    # If input was a scene, apply to its geometry too
    if isinstance(scene, trimesh.Scene):
        for g in scene.geometry.values():
            g.vertices *= scale
        out_path = os.path.join(args.output_dir, f"object_{args.obj_id}_scaled.glb")
        scene.export(out_path)
    else:
        out_path = os.path.join(args.output_dir, f"object_{args.obj_id}_scaled.glb")
        mesh.export(out_path)
    print(f"Scaled mesh saved to {out_path}")

if __name__ == "__main__":
    main()