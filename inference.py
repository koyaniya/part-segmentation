import os
import torch
import numpy as np
import open3d as o3d
import argparse
import trimesh
from pointcept.models.SAMPart3D import SAMPart3D
import torch.nn as nn
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser("SAMPart3D inference")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input PLY from previous pipeline step"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output segmented PLY"
    )
    return parser.parse_args()

args = parse_args()

INPUT_PLY = args.input
OUTPUT_PLY = args.output
CKPT_PATH = "/home/piai/SAMPart3D/ckpts/ptv3-object.pth"    # 경로 확인 필요
DEVICE = "cuda"


# LOAD POINT CLOUD
pcd = o3d.io.read_point_cloud(INPUT_PLY)


if not pcd.has_normals():
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=50
        )
    )

orig_points = np.asarray(pcd.points, dtype=np.float32)
points = orig_points.copy()
normals = np.asarray(pcd.normals, dtype=np.float32)

if pcd.has_colors():
    colors = np.asarray(pcd.colors, dtype=np.float32)
else:
    colors = np.ones_like(points) * 0.5



# NORMALIZE
center = points.mean(axis=0)
points = points - center
norm_scale = np.max(np.linalg.norm(points, axis=1))
points = points / norm_scale


# BUILD MODEL
model_cfg = dict(
    type="SAMPart3D",
    backbone_dim=384,
    output_dim=384,
    pcd_feat_dim=9,
    freeze_backbone=True,
    max_grouping_scale=2,
    use_hierarchy_losses=True,
    backbone=dict(
        type="PTv3-obj",
        in_channels=9,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(),
        enc_depths=(3, 3, 3, 6, 16),
        enc_channels=(32, 64, 128, 256, 384),
        enc_num_head=(2, 4, 8, 16, 24),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,))

model = build_model(model_cfg).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location="cpu")

if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=False)
model.eval()

import torch.nn as nn
model.quantile_transformer = nn.Identity()

# BUILD BATCH

coord = torch.as_tensor(points, dtype=torch.float32, device=DEVICE)
normal = torch.as_tensor(normals, dtype=torch.float32, device=DEVICE)
color = torch.as_tensor(colors, dtype=torch.float32, device=DEVICE)


MAX_POINTS = 20000   # 수정 가능 (GPU 의 성능에 따라 변경) 
subsample_idx = None

if coord.shape[0] > MAX_POINTS:
    subsample_idx = torch.randperm(coord.shape[0], device=DEVICE)[:MAX_POINTS]
    coord = coord[subsample_idx]
    normal = normal[subsample_idx]
    color = color[subsample_idx]
    
offset = torch.tensor(
    [coord.shape[0]],
    dtype=torch.long,
    device=DEVICE
)

feat = torch.cat([coord, normal, color], dim=1)   # (N, 9)

batch_idx = torch.zeros(
    coord.shape[0],
    dtype=torch.long,
    device=DEVICE
)


num_points = coord.shape[0]
scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)


coord = coord.contiguous()
feat = feat.contiguous()
batch_idx = batch_idx.contiguous()
offset = offset.contiguous()
scale = scale.contiguous()



batch = {
    "obj": {
        "coord": coord,
        "feat": feat,
        "batch": batch_idx,
        "offset": offset,
    },
    "scale": scale,
}


# INFERENCE
with torch.no_grad():
    print(
        coord.device,
        feat.device,
        batch_idx.device,
        offset.device,
        scale.device,
        next(model.parameters()).device
    )
    out = model(batch)
MIN_PART_POINTS = 300
pred = out.argmax(dim=1).cpu().numpy()


unique_labels, counts = np.unique(pred, return_counts=True)

MIN_PART_POINTS = 300  # 수정 가능

valid_parts = [
    int(lbl)
    for lbl, cnt in zip(unique_labels, counts)
    if cnt >= MIN_PART_POINTS
]

num_valid_parts = len(valid_parts)

print("All predicted labels:", unique_labels.tolist())
print("Points per label:", dict(zip(unique_labels.tolist(), counts.tolist())))
print("Valid parts (>= {} pts):".format(MIN_PART_POINTS), valid_parts)
print("Number of valid parts:", num_valid_parts)

if num_valid_parts < 3:
    print("SAMPart3D segmentation insufficient (<3 parts)")
    run_fallback = True
else:
    print("SAMPart3D segmentation OK")
    run_fallback = False

if not run_fallback:

    pcd_out = o3d.geometry.PointCloud()

    if subsample_idx is not None:
        save_points = orig_points[subsample_idx.cpu().numpy()]
    else:
        save_points = orig_points

    pcd_out.points = o3d.utility.Vector3dVector(save_points)

    colors = np.zeros((pred.shape[0], 3), dtype=np.float32)

    palette = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.0],
    ], dtype=np.float32)

    for i in range(pred.shape[0]):
        colors[i] = palette[pred[i] % len(palette)]

    pcd_out.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(OUTPUT_PLY, pcd_out)
    print("Saved segmented point cloud to:", OUTPUT_PLY)


if run_fallback = True:
    pcd = o3d.io.read_point_cloud("/home/piai/SAMPart3D/20_desk_gaussian.ply")
    points = np.asarray(pcd.points)
    print(pcd)
    print("Points:", points.shape)

    o3d.visualization.draw_geometries([pcd])

    #top
    plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=2000)

    [a, b, c, d] = plane_model
    print("Plane normal:", a, b, c, d)

    tabletop = pcd.select_by_index(inliers)

    o3d.io.write_point_cloud("part_top.ply", tabletop)
    top = o3d.io.read_point_cloud("/home/piai/SAMPart3D/part_top.ply")
    o3d.visualization.draw_geometries([top])

    #legs 
    
    thickness_leg = 0.0534 #수정 가능 
    thickness_frame = 0.03 #수정 가능 
    normal_norm = np.sqrt(a*a + b*b + c*c)
    distances = np.abs(points @ np.array([a, b, c]) + d) / normal_norm

    tabletop_mask_leg = distances < thickness_leg
    tabletop_leg = pcd.select_by_index(np.where(tabletop_mask_leg)[0])
    legs = pcd.select_by_index(np.where(~tabletop_mask_leg)[0])
    tabletop_mask_frame = distances < thickness_frame
    tabletop_frame = pcd.select_by_index(np.where(tabletop_mask_frame)[0])
    frame = pcd.select_by_index(np.where(~tabletop_mask_frame)[0])

    legs_clean, ind = legs.remove_radius_outlier(
    nb_points=30,  
    radius=0.02)


    o3d.visualization.draw_geometries([legs_clean])

    pts = np.asarray(legs_clean.points)

    X = pts[:, :2]   # cluster in XY

    db = DBSCAN(eps=0.05, min_samples=50)

    labels = db.fit_predict(X)

    unique_labels = [l for l in np.unique(labels) if l != -1]
    print("Clusters found:", unique_labels)


    for i in unique_labels:
        idx = np.where(labels == i)[0]
        part = legs_clean.select_by_index(idx)
        o3d.io.write_point_cloud(f"leg_{i+1}.ply", part)

    leg = o3d.io.read_point_cloud("/home/piai/SAMPart3D/leg_4.ply")
    o3d.visualization.draw_geometries([leg])



    #frame 

    o3d.visualization.draw_geometries([frame])


    frame_pts = np.asarray(frame.points)
    leg_files = ["leg_1.ply", "leg_2.ply", "leg_3.ply","leg_4.ply"]

    legs = []
    for f in leg_files: 
        leg = o3d.io.read_point_cloud(f)
        legs.append(leg)

    legs_pc = legs[0]
    for l in legs[1:]:
        legs_pc += l

    leg_pts = np.asarray(legs_pc.points)
    nbrs = NearestNeighbors(n_neighbors=1).fit(leg_pts)

    leg_dist, _ = nbrs.kneighbors(frame_pts)
    leg_clearance = 0.02   # adjust if needed (0.01–0.03 typical)

    keep_mask = leg_dist[:, 0] > leg_clearance
    frame_no_legs_pts = frame_pts[keep_mask]

    frame_no_legs = o3d.geometry.PointCloud()
    frame_no_legs.points = o3d.utility.Vector3dVector(frame_no_legs_pts)

    o3d.io.write_point_cloud("manual_output/frame_without_legs.ply", frame_no_legs)


    frame = o3d.io.read_point_cloud("/home/piai/SAMPart3D/frame_without_legs.ply")
    o3d.visualization.draw_geometries([frame])


    frame, _ = frame.remove_radius_outlier(nb_points=60,
    radius=0.01)

    o3d.visualization.draw_geometries([frame])

    points = np.asarray(frame.points)
    xy = points[:, :2] 

    db = DBSCAN(eps=0.025, min_samples=40).fit(xy)

    labels = db.labels_

    unique_labels = set(labels)
    print("Clusters:", unique_labels)

    frame_id = 1 


    for lbl in sorted(unique_labels):
        if lbl == -1:
            continue

        idx = np.where(labels == lbl)[0]
        cluster = frame.select_by_index(idx)

        print(f"Saving frame_{frame_id}.ply  (DBSCAN label={lbl}, points={len(idx)})")

        o3d.io.write_point_cloud(f"frame_{frame_id}.ply", cluster)

        frame_id += 1


    frame = o3d.io.read_point_cloud("/home/piai/SAMPart3D/frame_4.ply")
    o3d.visualization.draw_geometries([frame])