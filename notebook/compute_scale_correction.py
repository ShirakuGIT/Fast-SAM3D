#!/usr/bin/env python3
"""
compute_scale_correction.py

Computes the global scale correction factor using the first object (bottle)
by comparing the actual metric height from the depth pointcloud with the
height of the reconstructed rough metric mesh.
"""
import os, sys, json, argparse, numpy as np
import cv2
import trimesh

def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data['intrinsic_matrix']).reshape(3,3)

def unproject(depth, K, mask):
    """depth: (H,W) in metres, mask: (H,W) uint8, returns (N,3)"""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = mask > 128
    U = u[valid]
    V = v[valid]
    Z = depth[valid]
    X = (U - K[0,2]) * Z / K[0,0]
    Y = (V - K[1,2]) * Z / K[1,1]
    return np.stack([X, Y, Z], axis=-1)

def pca_height(verts):
    """Get the length of the major axis after PCA alignment."""
    center = verts.mean(axis=0)
    cov = np.cov((verts - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    aligned = (verts - center) @ eigvecs[:, order]
    return aligned[:,0].max() - aligned[:,0].min()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--obj_id", type=int, default=0)
    args = parser.parse_args()

    # Load real pointcloud (from depth + mask)
    mask_path = os.path.join(args.data_dir, f"{args.obj_id}_mask.png")
    depth_path = os.path.join(args.data_dir, f"{args.obj_id}_depth.png")
    intrinsics_path = os.path.join(args.data_dir, "cam_K.json")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth_m = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * 0.001
    K = load_intrinsics(intrinsics_path)
    pcd_real = unproject(depth_m, K, mask)
    real_height = pca_height(pcd_real)
    print(f"Real bottle height from depth: {real_height*100:.2f} cm")

    # Load rough metric mesh
    mesh_path = os.path.join(args.mesh_dir, f"object_{args.obj_id}_rough_metric.glb")
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        verts = np.concatenate([g.vertices for g in scene.geometry.values()], axis=0)
    else:
        verts = np.asarray(scene.vertices)
    mesh_height = pca_height(verts)
    print(f"Mesh height: {mesh_height*100:.2f} cm")

    factor = real_height / mesh_height if mesh_height != 0 else 1.0
    print(f"Correction factor: {factor:.4f}")

    # Save factor
    with open("scale_correction.txt", "w") as f:
        f.write(f"{factor}\n")
    print("Saved scale_correction.txt")

if __name__ == "__main__":
    main()