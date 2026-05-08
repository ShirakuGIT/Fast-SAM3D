#!/usr/bin/env python3
"""
refine_scale_bruteforce.py

Fully automatic metric scale recovery by trying all 24 right-handed coordinate
remappings between the mesh and the camera/depth frame. No manual calibration needed.

Usage:
    python refine_scale_bruteforce.py \
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

def all_right_handed_permutations():
    """
    Returns a list of 3x3 permutation/sign matrices representing all
    right-handed orthogonal coordinate transformations (row-major).
    """
    axes = np.array([[1,0,0], [-1,0,0],
                     [0,1,0], [0,-1,0],
                     [0,0,1], [0,0,-1]])
    perms = []
    from itertools import permutations
    for (i, j, k) in permutations(range(6), 3):
        # check that axes are orthogonal
        m = np.stack([axes[i], axes[j], axes[k]]).T  # 3x3
        if abs(np.linalg.det(m) - 1.0) < 1e-6:
            perms.append(m)
    return perms

def project_and_count(verts, K, H, W, depth_metric, mask, R, t, verbose=False):
    """
    Projects vertices after applying rotation R and translation t.
    Returns number of inlier points and the scale estimate from those inliers.
    """
    # Transform vertices: v_cam = (v @ R) + t   (assuming v row vectors)
    # Actually we'll apply rotation and translation to bring them into camera frame.
    v_trans = verts @ R + t   # (N,3)
    X, Y, Z = v_trans[:,0], v_trans[:,1], v_trans[:,2]
    # Filter Z>0 (in front of camera)
    front = Z > 1e-6
    if np.sum(front) == 0:
        return 0, None
    X, Y, Z = X[front], Y[front], Z[front]
    u = (X * K[0,0]) / Z + K[0,2]
    v = (Y * K[1,1]) / Z + K[1,2]
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)
    inside = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    if np.sum(inside) == 0:
        return 0, None
    u_i, v_i = u_int[inside], v_int[inside]
    Z_i = Z[inside]
    # Mask and depth check
    mask_vals = mask[v_i, u_i] > 128
    if np.sum(mask_vals) == 0:
        return 0, None
    u_m, v_m = u_i[mask_vals], v_i[mask_vals]
    Z_m = Z_i[mask_vals]
    real_depth = depth_metric[v_m, u_m]
    valid = (real_depth > 0) & (np.abs(real_depth - Z_m) < 0.02)  # 2 cm tolerance
    n_inliers = np.sum(valid)
    if n_inliers < 10:
        return n_inliers, None
    ratios = real_depth[valid] / Z_m[valid]
    scale = np.median(ratios)
    return n_inliers, scale

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
    H, W = depth_metric.shape

    print(f"Loading rough metric mesh: {mesh_path}")
    scene = trimesh.load(mesh_path)
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
        verts = np.concatenate(all_verts, axis=0)
    else:
        verts = np.asarray(scene.vertices)

    # Generate all axis mappings
    perms = all_right_handed_permutations()
    best_inliers = 0
    best_scale = None
    best_perm = None
    best_t = None

    # Try a range of possible translations (we can estimate by looking at the mesh centroid and a point in front)
    # For simplicity, we assume translation is zero (the mesh has already been roughly translated)
    # but we can also try the centroid offset.
    T_candidates = [np.zeros(3)]  # identity
    # Also try the negative centroid to bring mesh center to origin
    centroid = verts.mean(axis=0)
    T_candidates.append(-centroid)

    for R in perms:
        for t in T_candidates:
            n, scale = project_and_count(verts, K, H, W, depth_metric, mask, R, t)
            if n > best_inliers:
                best_inliers = n
                best_scale = scale
                best_perm = (R, t)
                print(f"New best: {best_inliers} inliers, scale={scale:.4f}")

    if best_scale is None:
        raise RuntimeError("Failed to find any alignment between mesh and depth. Check data.")

    print(f"Best alignment: {best_inliers} inliers, scale={best_scale:.4f}")
    # Apply the transformation and scaling to the original mesh
    R, t = best_perm
    # We need to transform vertices and then scale them
    new_verts = (verts @ R + t) * best_scale
    # Create output mesh
    out_mesh = trimesh.Trimesh(vertices=new_verts,
                               faces=all_faces if isinstance(scene, trimesh.Scene) else scene.faces)
    out_path = os.path.join(args.output_dir, f"object_{args.obj_id}_scaled.glb")
    out_mesh.export(out_path)
    print(f"Saved scaled mesh to {out_path}")

if __name__ == "__main__":
    main()