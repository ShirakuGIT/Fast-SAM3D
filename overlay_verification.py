import os, json, numpy as np, cv2, trimesh
import matplotlib.pyplot as plt

DATA_DIR = "../SceneComplete/gateway_jobs/20260504_184605_570af3/exp_20260504_184605/grasp_data"
MESH_DIR = "output_scene_final/objects"
POSE_DIR = "output_scene_final/poses"    # where you saved the ICP poses
OUTPUT   = "overlay_result.png"

# Load the RGB image (convert BGR → RGB for matplotlib)
rgb = cv2.imread(os.path.join(DATA_DIR, "scene_full_image.png"))[..., ::-1]

# Camera intrinsics
with open(os.path.join(DATA_DIR, "cam_K.json")) as f:
    K = np.array(json.load(f)['intrinsic_matrix']).reshape(3, 3)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Colours for the four objects
colours = ['red', 'lime', 'cyan', 'yellow']

plt.figure(figsize=(12, 8))
plt.imshow(rgb)

for i in range(4):
    mesh_path = os.path.join(MESH_DIR, f"object_{i}_true_metric.glb")
    pose_path = os.path.join(POSE_DIR, f"object_{i}_cam_pose.npy")
    mask_path = os.path.join(DATA_DIR, f"{i}_mask.png")

    if not os.path.exists(mesh_path):
        continue

    # Load mesh (already in camera frame, metric)
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        verts = np.concatenate([g.vertices for g in scene.geometry.values()], axis=0)
    else:
        verts = np.asarray(scene.vertices)

    # Apply the ICP refinement (small correction)
    if os.path.exists(pose_path):
        T = np.load(pose_path)
        verts_hom = np.column_stack([verts, np.ones(len(verts))])
        verts = (T @ verts_hom.T).T[:, :3]

    # Project vertices onto the image
    X, Y, Z = verts[:, 0], verts[:, 1], verts[:, 2]
    front = Z > 0
    u = (X[front] * fx) / Z[front] + cx
    v = (Y[front] * fy) / Z[front] + cy

    # Keep only points inside the image boundaries
    inside = (u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0])
    u, v = u[inside], v[inside]

    # Draw projected mesh points
    plt.scatter(u, v, s=1, c=colours[i], label=f'obj {i} (mesh)', alpha=0.6)

    # Draw mask contour (if available)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=colours[i], linewidth=1.5,
                         linestyle='--', label=f'obj {i} (mask)' if cnt is contours[0] else "")

plt.legend(loc='lower right', fontsize='small', framealpha=0.9)
plt.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"Saved {OUTPUT}")