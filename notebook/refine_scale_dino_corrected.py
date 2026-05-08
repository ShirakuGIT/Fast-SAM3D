#!/usr/bin/env python3
"""
DINO‑based scaling refinement – identical to SceneComplete’s method.
Renders the SAM‑3D mesh with its native vertex colours, finds dense
DINOv2 correspondences with the real masked image, and computes the
scale factor from point‑cloud alignment.

Usage:
    python refine_scale_dino_corrected.py \
        --data_dir ../SceneComplete/.../grasp_data \
        --mesh_dir output_scene_final/objects \
        --obj_id 0 \
        --output_dir output_scene_final/objects/scaled
"""

import os, sys, json, argparse, numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
import open3d as o3d
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

# ---------- DINOv2 feature extractor ----------
class DINOv2FeatureExtractor:
    def __init__(self, model_name="facebook/dinov2-small", device="cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size = self.model.config.patch_size

    @torch.no_grad()
    def extract_features(self, pil_img):
        """Returns feature map (C, H_p, W_p) for a PIL image."""
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state[:, 1:, :]  # skip CLS
        B, N, D = feats.shape
        H_p = int(pil_img.size[1] // self.patch_size)
        W_p = int(pil_img.size[0] // self.patch_size)
        feats = feats.permute(0, 2, 1).reshape(B, D, H_p, W_p)
        return feats.squeeze(0)

# ---------- Mutual nearest neighbour correspondences ----------
def find_correspondences(feats1, feats2, num_pairs=20, thresh=0.05):
    C, H1, W1 = feats1.shape
    _, H2, W2 = feats2.shape
    f1 = feats1.reshape(C, -1)
    f2 = feats2.reshape(C, -1)
    f1 = F.normalize(f1, p=2, dim=0)
    f2 = F.normalize(f2, p=2, dim=0)
    sim = torch.mm(f1.T, f2)
    nn12 = sim.max(dim=1)[1]
    nn21 = sim.max(dim=0)[1]
    mutual = nn21[nn12] == torch.arange(len(nn12), device=sim.device)
    idx1 = torch.where(mutual)[0].cpu().numpy()
    idx2 = nn12[idx1].cpu().numpy()
    rows1, cols1 = idx1 // W1, idx1 % W1
    rows2, cols2 = idx2 // W2, idx2 % W2
    coord1 = np.stack([rows1, cols1], axis=-1)
    coord2 = np.stack([rows2, cols2], axis=-1)
    if len(coord1) > num_pairs:
        rng = np.random.default_rng(seed=42)
        select = rng.choice(len(coord1), num_pairs, replace=False)
        coord1 = coord1[select]
        coord2 = coord2[select]
    return coord1, coord2

# ---------- Camera helpers ----------
def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data['intrinsic_matrix']).reshape(3, 3)

def back_project(pixel_coords, depth_map, K):
    points = []
    for r, c in pixel_coords:
        r, c = int(round(r)), int(round(c))
        if r < 0 or r >= depth_map.shape[0] or c < 0 or c >= depth_map.shape[1]:
            continue
        Z = depth_map[r, c]
        if Z <= 0:
            continue
        X = (c - K[0,2]) * Z / K[0,0]
        Y = (r - K[1,2]) * Z / K[1,1]
        points.append([X, Y, Z])
    return np.array(points)

def estimate_similarity_transform(src, tgt):
    centroid_s = np.mean(src, axis=0)
    centroid_t = np.mean(tgt, axis=0)
    centered_s = src - centroid_s
    centered_t = tgt - centroid_t
    scale = np.linalg.norm(centered_t) / np.linalg.norm(centered_s) if np.linalg.norm(centered_s) > 1e-8 else 1.0
    scaled_s = centered_s * scale
    H = scaled_s.T @ centered_t
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_t - R @ (centroid_s * scale)
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    return T, scale

# ---------- Mesh rendering WITH VERTEX COLOURS ----------
def render_mesh_colored(scene_or_mesh, K, width, height):
    """
    Render the mesh from the camera origin with identity extrinsic,
    using the mesh's native vertex colours.
    Returns RGB (uint8 H,W,3) and depth (float32 H,W in metres).
    """
    # Merge if scene
    if isinstance(scene_or_mesh, trimesh.Scene):
        verts_list, faces_list, colors_list = [], [], []
        offset = 0
        for g in scene_or_mesh.geometry.values():
            v = np.asarray(g.vertices)
            f = np.asarray(g.faces) + offset
            # Get vertex colours: if not present, fallback to white
            if hasattr(g.visual, 'vertex_colors'):
                c = g.visual.vertex_colors[:, :3] / 255.0   # assume 0-255
            else:
                c = np.ones((len(v), 3))
            verts_list.append(v)
            faces_list.append(f)
            colors_list.append(c)
            offset += len(v)
        verts = np.concatenate(verts_list)
        faces = np.concatenate(faces_list)
        colors = np.concatenate(colors_list)
    else:
        verts = np.asarray(scene_or_mesh.vertices)
        faces = np.asarray(scene_or_mesh.faces)
        if hasattr(scene_or_mesh.visual, 'vertex_colors'):
            colors = scene_or_mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = np.ones((len(verts), 3))

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    renderer.setup_camera(intrinsic, np.eye(4))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", o3d_mesh, mat)
    renderer.scene.set_background([0, 0, 0, 0])

    rgb = np.asarray(renderer.render_to_image())
    depth = np.asarray(renderer.render_to_depth_image())
    renderer.scene.remove_geometry("mesh")
    return rgb, depth

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--obj_id", type=int, default=0)
    parser.add_argument("--output_dir", default="./scaled")
    parser.add_argument("--num_pairs", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    mesh_dir = args.mesh_dir
    obj_id = args.obj_id
    os.makedirs(args.output_dir, exist_ok=True)

    masked_rgb_path = os.path.join(data_dir, f"{obj_id}_masked.png")
    depth_path = os.path.join(data_dir, f"{obj_id}_depth.png")
    intrinsics_path = os.path.join(data_dir, "cam_K.json")
    mesh_path = os.path.join(mesh_dir, f"object_{obj_id}_rough_metric.glb")

    print(f"Loading mesh: {mesh_path}")
    scene_or_mesh = trimesh.load(mesh_path)

    real_rgb = cv2.imread(masked_rgb_path, cv2.IMREAD_COLOR)
    if real_rgb is None:
        raise FileNotFoundError(f"Could not read {masked_rgb_path}")
    real_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * 0.001
    H, W = real_depth.shape
    K = load_intrinsics(intrinsics_path)

    # Render with vertex colours
    print("Rendering mesh with vertex colours...")
    synth_rgb, synth_depth = render_mesh_colored(scene_or_mesh, K, W, H)

    if args.debug:
        cv2.imwrite("debug_synth_rgb.png", cv2.cvtColor(synth_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_synth_depth.png", (synth_depth * 1000).astype(np.uint16))
        print("Saved debug images.")

    # DINO matching
    load_size = 224
    synth_pil = Image.fromarray(synth_rgb)
    real_pil = Image.fromarray(cv2.cvtColor(real_rgb, cv2.COLOR_BGR2RGB))
    synth_resized = synth_pil.resize((load_size, load_size), Image.BILINEAR)
    real_resized = real_pil.resize((load_size, load_size), Image.BILINEAR)

    print("Extracting DINOv2 features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = DINOv2FeatureExtractor(device=device)
    feats_synth = dino.extract_features(synth_resized)
    feats_real = dino.extract_features(real_resized)

    print("Finding correspondences...")
    pts1_feat, pts2_feat = find_correspondences(feats_synth, feats_real, num_pairs=args.num_pairs)

    # Map feature grid coords to original image coords
    patch_s = dino.patch_size
    pts1_resized = pts1_feat * patch_s
    pts2_resized = pts2_feat * patch_s
    scale_row = H / load_size
    scale_col = W / load_size
    pts1_orig = pts1_resized * np.array([scale_row, scale_col])
    pts2_orig = pts2_resized * np.array([scale_row, scale_col])

    if args.debug:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(synth_rgb)
        axes[0].scatter(pts1_orig[:, 1], pts1_orig[:, 0], c='red', s=30)
        axes[0].set_title("Synthetic render (vertex colours)")
        axes[1].imshow(cv2.cvtColor(real_rgb, cv2.COLOR_BGR2RGB))
        axes[1].scatter(pts2_orig[:, 1], pts2_orig[:, 0], c='red', s=30)
        axes[1].set_title("Real masked image")
        plt.savefig("correspondences.png")
        plt.close()
        print("Saved correspondences.png")

    # Back‑project
    synth_3d = back_project(pts1_orig, synth_depth, K)
    real_3d = back_project(pts2_orig, real_depth, K)

    if len(synth_3d) < 3 or len(real_3d) < 3:
        print("⚠️ Not enough valid correspondences – could not compute scale.")
        return

    _, scale_factor = estimate_similarity_transform(synth_3d, real_3d)
    print(f"Estimated scale factor: {scale_factor:.4f}")

    # Apply scale
    if isinstance(scene_or_mesh, trimesh.Scene):
        for g in scene_or_mesh.geometry.values():
            g.vertices *= scale_factor
        out_path = os.path.join(args.output_dir, f"object_{obj_id}_scaled.glb")
        scene_or_mesh.export(out_path)
    else:
        scene_or_mesh.vertices *= scale_factor
        out_path = os.path.join(args.output_dir, f"object_{obj_id}_scaled.glb")
        scene_or_mesh.export(out_path)
    print(f"Saved scaled mesh to {out_path}")

if __name__ == "__main__":
    main()