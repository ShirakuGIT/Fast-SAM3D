#!/usr/bin/env python3
"""
refine_scale_scenecomplete.py

Replicates the SceneComplete mesh scaling pipeline:
1. Render mesh from camera view → synthetic RGB+depth
2. Extract DINO-ViT feature correspondences (synthetic ↔ real masked image)
3. Back-project correspondence pixels to 3D using depth + intrinsics
4. Estimate similarity transform → extract uniform scale factor
5. Apply scale to mesh vertices

Usage:
    python refine_scale_scenecomplete.py \
        --data_dir ../SceneComplete/gateway_jobs/20260504_184605_570af3/exp_20260504_184605/grasp_data \
        --mesh_dir output_scene_final/objects \
        --obj_id 0 \
        --output_dir output_scene_final/objects/scaled \
        --debug
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
import open3d as o3d
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.spatial.transform import Rotation

# ============================================================================
# DINO-ViT Feature Extraction (matching SceneComplete's implementation)
# ============================================================================

def load_dino_model(model_type='dino_vits8', device='cuda'):
    """Load DINO ViT-S/8 model (same as SceneComplete)."""
    try:
        # Try torch.hub first (SceneComplete style)
        model = torch.hub.load('facebookresearch/dino:main', model_type, pretrained=True)
    except:
        # Fallback to transformers
        from transformers import AutoModel, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        return model, processor, "dinov2"
    
    model.eval().to(device)
    return model, None, "dino"

def extract_dino_features(model, pil_img, load_size=224, layer=9, facet='key', device='cuda', model_type="dino"):
    """
    Extract DINO features matching SceneComplete's approach.
    Returns: feature map (C, H_patch, W_patch)
    """
    if model_type == "dinov2":
        # Fallback path for transformers-based loading
        inputs = model_type.processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 1:, :]  # skip CLS
        B, N, D = feats.shape
        patch_size = 14  # dinov2-small patch size
        H_p = int(pil_img.size[1] // patch_size)
        W_p = int(pil_img.size[0] // patch_size)
        feats = feats.permute(0, 2, 1).reshape(B, D, H_p, W_p)
        return feats.squeeze(0)
    
    # Original DINO ViT-S/8 path
    transform = transforms.Compose([
        transforms.Resize(load_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get attention weights and features
        features = model.get_intermediate_layers(img_tensor, n=1)[0]  # (B, N+1, D)
        features = features[:, 1:, :]  # remove CLS token: (B, N, D)
        
        # Reshape to spatial feature map
        B, N, D = features.shape
        patch_size = 8  # ViT-S/8
        H_p = int(pil_img.size[1] // patch_size)
        W_p = int(pil_img.size[0] // patch_size)
        
        if facet == 'key':
            # For key features, we use the raw output
            feats = features.permute(0, 2, 1).reshape(B, D, H_p, W_p)
        else:
            feats = features.permute(0, 2, 1).reshape(B, D, H_p, W_p)
    
    return feats.squeeze(0)  # (C, H_p, W_p)


def find_correspondences_dino(feats1, feats2, num_pairs=15, thresh=0.05):
    """
    Find mutual nearest neighbor correspondences between two feature maps.
    Matches SceneComplete's implementation exactly.
    
    Args:
        feats1, feats2: torch tensors of shape (C, H, W)
        num_pairs: number of correspondence pairs to return
        thresh: similarity threshold (unused in MNN, kept for API compatibility)
    
    Returns:
        coords1, coords2: numpy arrays of shape (N, 2) with (row, col) in feature grid
    """
    C, H1, W1 = feats1.shape
    _, H2, W2 = feats2.shape
    
    # Flatten and normalize
    f1 = feats1.reshape(C, -1)  # (C, N1)
    f2 = feats2.reshape(C, -1)  # (C, N2)
    f1 = F.normalize(f1, p=2, dim=0)
    f2 = F.normalize(f2, p=2, dim=0)
    
    # Compute similarity matrix
    sim = torch.mm(f1.T, f2)  # (N1, N2)
    
    # Mutual nearest neighbors
    nn12 = sim.max(dim=1)[1]  # for each f1, best match in f2
    nn21 = sim.max(dim=0)[1]  # for each f2, best match in f1
    mutual = nn21[nn12] == torch.arange(len(nn12), device=sim.device)
    
    idx1 = torch.where(mutual)[0].cpu().numpy()
    idx2 = nn12[idx1].cpu().numpy()
    
    # Convert flat indices to (row, col) in feature grid
    rows1 = idx1 // W1
    cols1 = idx1 % W1
    rows2 = idx2 // W2
    cols2 = idx2 % W2
    
    coords1 = np.stack([rows1, cols1], axis=-1)  # (N, 2)
    coords2 = np.stack([rows2, cols2], axis=-1)
    
    # Subsample if needed
    if len(coords1) > num_pairs:
        rng = np.random.default_rng(seed=42)
        select = rng.choice(len(coords1), num_pairs, replace=False)
        coords1 = coords1[select]
        coords2 = coords2[select]
    
    return coords1, coords2


# ============================================================================
# Pointcloud & Projection Utilities (matching SceneComplete exactly)
# ============================================================================

def parse_intrinsics(intrinsics_dict):
    """Extract fx, fy, cx, cy from intrinsics dict (SceneComplete format)."""
    K = np.array(intrinsics_dict['intrinsic_matrix']).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return cx, cy, fx, fy


def project_depth_to_camera_space(pixels, K):
    """
    Project pixel coordinates + depth to 3D camera space.
    pixels: (N, 3) array of [row, col, depth_meters]
    Returns: (N, 3) array of [X, Y, Z] in camera coordinates
    """
    rows = pixels[:, 0]  # v
    cols = pixels[:, 1]  # u
    Z = pixels[:, 2]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    X = (cols - cx) * Z / fx
    Y = (rows - cy) * Z / fy
    
    return np.stack([X, Y, Z], axis=1)


def build_pointcloud_from_depth(depth_path, rgb_path, intrinsics_path, remove_origin=False):
    """
    Build Open3D pointcloud from depth + RGB images (SceneComplete style).
    
    Returns:
        pcd: o3d.geometry.PointCloud
        points_array: np.ndarray of shape (N, 3) with 3D points
        colors_array: np.ndarray of shape (N, 3) with RGB colors [0,1]
    """
    # Load images
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    
    # Convert depth mm -> meters
    depth_m = depth * 0.001
    
    H, W = depth.shape
    
    # Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics['intrinsic_matrix']).reshape(3, 3)
    
    # Create meshgrid of pixel coordinates (row-major order)
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels_flat = np.stack([
        rows.ravel(),      # row (v)
        cols.ravel(),      # col (u)
        depth_m.ravel()    # depth in meters
    ], axis=1)  # (H*W, 3)
    
    # Project to 3D
    points_3d = project_depth_to_camera_space(pixels_flat, K)
    
    # Extract colors (normalized to [0, 1])
    colors_flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    
    # Filter origin points if requested
    if remove_origin:
        valid_mask = np.linalg.norm(points_3d, axis=1) > 1e-6
        points_3d = points_3d[valid_mask]
        colors_flat = colors_flat[valid_mask]
    
    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)
    
    return pcd, points_3d, colors_flat, K, H, W


def get_pixel_indices_in_pointcloud(points_array, colors_array, pixel_coords, H, W):
    """
    Get 3D points corresponding to pixel coordinates in a flattened pointcloud.
    Matches SceneComplete's row-major indexing exactly.
    
    Args:
        points_array: (N, 3) array of 3D points
        colors_array: (N, 3) array of colors
        pixel_coords: list of (row, col) tuples
        H, W: image dimensions
    
    Returns:
        indices: list of indices in the flattened array
        values: list of corresponding 3D points
    """
    indices = []
    values = []
    
    for row, col in pixel_coords:
        row, col = int(round(row)), int(round(col))
        if 0 <= row < H and 0 <= col < W:
            idx = row * W + col  # row-major flattening
            if idx < len(points_array):
                indices.append(idx)
                values.append(points_array[idx])
    
    return indices, values


def is_origin_point(point, eps=1e-6):
    """Check if a 3D point is effectively at the origin."""
    return np.linalg.norm(point) < eps

def estimate_similarity_transform(source_pts, target_pts):
    """
    Estimate similarity transform (uniform scale + rotation + translation).
    Returns 4x4 transform matrix and scale factor.
    """
    # Safety check
    if len(source_pts) < 3 or len(target_pts) < 3:
        raise ValueError(f"Need at least 3 points, got {len(source_pts)} source and {len(target_pts)} target")
    
    centroid_s = np.mean(source_pts, axis=0)
    centroid_t = np.mean(target_pts, axis=0)
    
    centered_s = source_pts - centroid_s
    centered_t = target_pts - centroid_t
    
    # Compute scale
    norm_s = np.linalg.norm(centered_s)
    norm_t = np.linalg.norm(centered_t)
    
    if norm_s < 1e-8:
        print(f"  ⚠ Source points are degenerate (norm={norm_s:.2e}), using fallback scale")
        return np.eye(4), 1.0
    
    scale = norm_t / norm_s
    
    # Scale source
    scaled_s = centered_s * scale
    
    # Compute rotation via SVD
    H_mat = scaled_s.T @ centered_t
    U, _, Vt = np.linalg.svd(H_mat)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_t - R @ (centroid_s * scale)
    
    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    
    return T, scale


# ============================================================================
# Similarity Transform Estimation (exact SceneComplete implementation)
# ============================================================================

def estimate_similarity_transform(source_pts, target_pts):
    """
    Estimate similarity transform (uniform scale + rotation + translation).
    Returns 4x4 transform matrix and scale factor.
    
    Matches SceneComplete's implementation exactly.
    """
    centroid_s = np.mean(source_pts, axis=0)
    centroid_t = np.mean(target_pts, axis=0)
    
    centered_s = source_pts - centroid_s
    centered_t = target_pts - centroid_t
    
    # Compute scale
    norm_s = np.linalg.norm(centered_s)
    norm_t = np.linalg.norm(centered_t)
    scale = norm_t / norm_s if norm_s > 1e-8 else 1.0
    
    # Scale source
    scaled_s = centered_s * scale
    
    # Compute rotation via SVD
    H_mat = scaled_s.T @ centered_t
    U, _, Vt = np.linalg.svd(H_mat)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_t - R @ (centroid_s * scale)
    
    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    
    return T, scale


# ============================================================================
# Mesh Rendering (Open3D offscreen - matches your existing setup)
# ============================================================================

def render_mesh_trimesh(mesh_or_scene, K, width, height):
    """
    Render mesh using Open3D offscreen renderer.
    Returns RGB (uint8) and depth (float32 in meters).
    """
    # Handle trimesh Scene -> merge geometries
    if isinstance(mesh_or_scene, trimesh.Scene):
        verts_list, faces_list = [], []
        offset = 0
        for geom in mesh_or_scene.geometry.values():
            v = np.asarray(geom.vertices).copy()
            f = np.asarray(geom.faces).copy() + offset
            verts_list.append(v)
            faces_list.append(f)
            offset += len(v)
        if not verts_list:
            raise ValueError("Scene is empty")
        mesh = trimesh.Trimesh(vertices=np.concatenate(verts_list),
                               faces=np.concatenate(faces_list))
    else:
        mesh = mesh_or_scene
    
    # Ensure mesh has normals for proper lighting
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()
    
    # Convert to Open3D (with .copy() to avoid read-only array issues)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces).copy())
    
    # Fix: copy normals to ensure writability
    normals = np.asarray(mesh.vertex_normals).copy()
    if len(normals) > 0:
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Setup renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    renderer.setup_camera(intrinsic, np.eye(4))
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", o3d_mesh, mat)
    renderer.scene.set_background([0, 0, 0, 0])
    
    # Render
    rgb = np.asarray(renderer.render_to_image())
    depth = np.asarray(renderer.render_to_depth_image())  # already in meters
    
    renderer.scene.remove_geometry("mesh")
    return rgb, depth


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SceneComplete-style mesh scaling")
    parser.add_argument("--data_dir", required=True, help="Directory with segmented RGB, depth, intrinsics")
    parser.add_argument("--mesh_dir", required=True, help="Directory with rough metric meshes")
    parser.add_argument("--obj_id", type=int, default=0)
    parser.add_argument("--output_dir", default="./scaled")
    parser.add_argument("--num_pairs", type=int, default=15, help="Number of correspondence pairs")
    parser.add_argument("--fallback_scale", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dino_model", type=str, default='dino_vits8', 
                       choices=['dino_vits8', 'dino_vitb8', 'dinov2-small'])
    args = parser.parse_args()
    
    # Paths
    real_rgb_path = os.path.join(args.data_dir, f"{args.obj_id}_masked.png")
    real_depth_path = os.path.join(args.data_dir, f"{args.obj_id}_depth.png")
    intrinsics_path = os.path.join(args.data_dir, "cam_K.json")
    mesh_path = os.path.join(args.mesh_dir, f"object_{args.obj_id}_rough_metric.glb")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[1/7] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    print(f"[2/7] Loading real data")
    real_rgb = cv2.imread(real_rgb_path, cv2.IMREAD_COLOR)
    if real_rgb is None:
        raise FileNotFoundError(f"Could not read {real_rgb_path}")
    real_rgb = cv2.cvtColor(real_rgb, cv2.COLOR_BGR2RGB)
    
    real_depth = cv2.imread(real_depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    real_depth_m = real_depth * 0.001  # mm -> meters
    H, W = real_depth.shape
    
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    K = np.array(intrinsics['intrinsic_matrix']).reshape(3, 3)
    
    print(f"[3/7] Rendering synthetic view")
    synth_rgb, synth_depth = render_mesh_trimesh(mesh, K, W, H)
    
    if args.debug:
        cv2.imwrite("debug_synth_rgb.png", cv2.cvtColor(synth_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_synth_depth.png", (synth_depth * 1000).astype(np.uint16))
    
    print(f"[4/7] Extracting DINO features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # Load DINO model
    model, processor, model_type = load_dino_model(args.dino_model, device)
    
    # Prepare PIL images (resize to 224 for DINO)
    load_size = 224
    synth_pil = Image.fromarray(synth_rgb).resize((load_size, load_size), Image.BILINEAR)
    real_pil = Image.fromarray(real_rgb).resize((load_size, load_size), Image.BILINEAR)
    
    # Extract features
    feats_synth = extract_dino_features(model, synth_pil, load_size=load_size, 
                                        layer=9, facet='key', device=device, model_type=model_type)
    feats_real = extract_dino_features(model, real_pil, load_size=load_size,
                                       layer=9, facet='key', device=device, model_type=model_type)
    
    print(f"[5/7] Finding correspondences")
    coords_synth_feat, coords_real_feat = find_correspondences_dino(
        feats_synth, feats_real, num_pairs=args.num_pairs
    )
    print(f"  Found {len(coords_synth_feat)} mutual correspondences")
    
    if args.debug and len(coords_synth_feat) > 0:
        # Visualize correspondences
        patch_size = 8 if 'vits8' in args.dino_model else 14
        synth_px = coords_synth_feat * patch_size
        real_px = coords_real_feat * patch_size
        
        # Scale to original image size
        scale_r = H / load_size
        scale_c = W / load_size
        synth_px_orig = synth_px * np.array([scale_r, scale_c])
        real_px_orig = real_px * np.array([scale_r, scale_c])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(synth_rgb)
        axes[0].scatter(synth_px_orig[:, 1], synth_px_orig[:, 0], c='red', s=20)
        axes[0].set_title("Synthetic Render")
        axes[1].imshow(real_rgb)
        axes[1].scatter(real_px_orig[:, 1], real_px_orig[:, 0], c='red', s=20)
        axes[1].set_title("Real Masked Image")
        plt.tight_layout()
        plt.savefig("correspondences_sc.png", dpi=150)
        plt.close()
        print("  Saved correspondence visualization to correspondences_sc.png")
    
        print(f"[6/7] Building pointclouds and extracting 3D correspondences")
    
    # Convert feature-grid coords to original image pixel coords
    patch_size = 8 if 'vits8' in args.dino_model else 14
    synth_px_resized = coords_synth_feat * patch_size
    real_px_resized = coords_real_feat * patch_size
    
    scale_row = H / load_size
    scale_col = W / load_size
    synth_px_orig = synth_px_resized * np.array([scale_row, scale_col])
    real_px_orig = real_px_resized * np.array([scale_row, scale_col])
    
    # Extract 3D points at correspondence locations by direct projection
        # Extract 3D points at correspondence locations by direct projection
    synth_3d_pts = []
    real_3d_pts = []
    
    for (r1, c1), (r2, c2) in zip(synth_px_orig, real_px_orig):
        r1, c1 = int(round(r1)), int(round(c1))
        r2, c2 = int(round(r2)), int(round(c2))
        
        # Check bounds and valid depth for BOTH sides
        synth_valid = (0 <= r1 < H and 0 <= c1 < W and synth_depth[r1, c1] > 1e-6)
        real_valid = (0 <= r2 < H and 0 <= c2 < W and real_depth_m[r2, c2] > 1e-6)
        
        if synth_valid and real_valid:
            # Project synthetic side
            Z1 = synth_depth[r1, c1]
            X1 = (c1 - K[0,2]) * Z1 / K[0,0]
            Y1 = (r1 - K[1,2]) * Z1 / K[1,1]
            
            # Project real side
            Z2 = real_depth_m[r2, c2]
            X2 = (c2 - K[0,2]) * Z2 / K[0,0]
            Y2 = (r2 - K[1,2]) * Z2 / K[1,1]
            
            synth_3d_pts.append([X1, Y1, Z1])
            real_3d_pts.append([X2, Y2, Z2])
    
    synth_3d = np.array(synth_3d_pts) if synth_3d_pts else np.empty((0, 3))
    real_3d = np.array(real_3d_pts) if real_3d_pts else np.empty((0, 3))
    
    print(f"  Raw 3D correspondences: {len(synth_3d)}")
    
    # Filter origin/near-zero points (SceneComplete style) - VECTORIZED
    if len(synth_3d) > 0 and len(real_3d) > 0:
        synth_norms = np.linalg.norm(synth_3d, axis=1)
        real_norms = np.linalg.norm(real_3d, axis=1)
        valid_mask = (synth_norms > 1e-6) & (real_norms > 1e-6)
        synth_3d = synth_3d[valid_mask]
        real_3d = real_3d[valid_mask]
    
    print(f"  Valid 3D correspondences after filtering: {len(synth_3d)}")
    
    print(f"[7/7] Estimating scale factor")
    if len(synth_3d) >= 3:
        _, scale_factor = estimate_similarity_transform(synth_3d, real_3d)
        print(f"  ✓ DINO-based scale factor: {scale_factor:.4f}")
    else:
        scale_factor = args.fallback_scale
        print(f"  ⚠ Not enough valid correspondences, using fallback: {scale_factor:.4f}")
    
    # Apply scale to mesh
    print(f"Applying scale {scale_factor:.4f} to mesh...")
    if isinstance(mesh, trimesh.Scene):
        for geom in mesh.geometry.values():
            geom.vertices *= scale_factor
    else:
        mesh.vertices *= scale_factor
    
    # Save output
    out_path = os.path.join(args.output_dir, f"object_{args.obj_id}_scaled.glb")
    if isinstance(mesh, trimesh.Scene):
        mesh.export(out_path)
    else:
        mesh.export(out_path)
    
    print(f"✓ Saved scaled mesh to {out_path}")
    print(f"✓ Done! Scale factor: {scale_factor:.4f}")
    
    return scale_factor


if __name__ == "__main__":
    main()