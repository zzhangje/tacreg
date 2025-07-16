from dataclasses import dataclass
from typing import Tuple
import numpy as np
import open3d as o3d
import copy
import networkx as nx
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed


@dataclass
class TacRegParam:
    # downsample
    voxel_size: float = 0.001

    # normal estimation
    estimate_tar_normals: bool = True
    estimate_src_normals: bool = True
    normal_radius: float = 0.01
    normal_max_nn: int = 30

    # ISS keypoints
    iss_salient_radius: float = 0.004
    iss_non_max_radius: float = 0.003
    iss_gamma_21: float = 0.975
    iss_gamma_32: float = 0.975
    iss_min_neighbors: int = 5

    # FPFH features
    fpfh_radius: float = 0.005
    fpfh_max_nn: int = 100

    # correspondence
    correspondence_size: int = 500

    # graph parameters
    dist_threshold: float = 0.008
    angle_threshold: float = np.radians(30)

    # verification parameters
    candidate_size: int = 20000
    parallel_jobs: int = 4

    # debug
    verbose: bool = False


def fpfh_correspondence(
    src_pcd: o3d.geometry.PointCloud,
    tar_pcd: o3d.geometry.PointCloud,
    params: TacRegParam = TacRegParam(),
) -> Tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    np.ndarray,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
]:
    """
    Pre-process source and target point clouds by downsampling and estimating normals.
    Args:
        src_pcd (o3d.geometry.PointCloud): Source point cloud.
        tar_pcd (o3d.geometry.PointCloud): Target point cloud.
        params (FPFHParam): Parameters for pre-processing.
    Returns:
        tuple: Tuple containing the source ISS keypoints, target ISS keypoints,
               edge list of correspondences, source point cloud with normals,
               and target point cloud with normals.
    """
    # downsample point clouds
    tar_pcd = tar_pcd.voxel_down_sample(voxel_size=params.voxel_size)
    src_pcd = src_pcd.voxel_down_sample(voxel_size=params.voxel_size)

    # estimate normals and FPFH features
    if params.estimate_tar_normals:
        tar_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.normal_radius, max_nn=params.normal_max_nn
            )
        )
    else:
        print("Warning: TacReg does not estimate normals for target point cloud.")
    if params.estimate_src_normals:
        src_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.normal_radius, max_nn=params.normal_max_nn
            )
        )
    else:
        print("Warning: TacReg does not estimate normals for source point cloud.")

    tar_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tar_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.fpfh_radius, max_nn=params.fpfh_max_nn
        ),
    )
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.fpfh_radius, max_nn=params.fpfh_max_nn
        ),
    )

    # find ISS keypoints
    tar_iss = o3d.geometry.keypoint.compute_iss_keypoints(
        tar_pcd,
        salient_radius=params.iss_salient_radius,
        non_max_radius=params.iss_non_max_radius,
        gamma_21=params.iss_gamma_21,
        gamma_32=params.iss_gamma_32,
        min_neighbors=params.iss_min_neighbors,
    )
    src_iss = o3d.geometry.keypoint.compute_iss_keypoints(
        src_pcd,
        salient_radius=params.iss_salient_radius,
        non_max_radius=params.iss_non_max_radius,
        gamma_21=params.iss_gamma_21,
        gamma_32=params.iss_gamma_32,
        min_neighbors=params.iss_min_neighbors,
    )
    tar_iss_idx = np.arange(len(tar_iss.points))
    src_iss_idx = np.arange(len(src_iss.points))
    tar_tree = o3d.geometry.KDTreeFlann(tar_pcd)
    src_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i in range(len(tar_iss_idx)):
        [_, idx, _] = tar_tree.search_knn_vector_3d(tar_iss.points[i], 1)
        tar_iss_idx[i] = idx[0]
    for i in range(len(src_iss_idx)):
        [_, idx, _] = src_tree.search_knn_vector_3d(src_iss.points[i], 1)
        src_iss_idx[i] = idx[0]
    if params.verbose:
        print(
            f"[ISS] Source: {len(src_iss_idx)} points, Target: {len(tar_iss_idx)} points"
        )

    np_src_fpfh = src_fpfh.data.T[src_iss_idx]
    np_tar_fpfh = tar_fpfh.data.T[tar_iss_idx]
    fpfh_sim_mat = np.mean(
        np.abs(np_src_fpfh[:, None, :] - np_tar_fpfh[None, :, :]), axis=2
    )
    closest_tar_idx = np.argsort(fpfh_sim_mat, axis=1)[
        :, : max(params.correspondence_size // len(src_iss_idx), 1)
    ]
    edge_list = []
    for src_idx, tar_indices in enumerate(closest_tar_idx):
        for tar_idx in tar_indices:
            edge_list.append(np.array((src_idx, tar_idx)))

    if params.verbose:
        print(f"[Edge] Found {len(edge_list)} correspondences")

    return src_iss, tar_iss, np.array(edge_list), src_pcd, tar_pcd


def svd_estimate_transform_normal(src_points, tar_points, src_normals, tar_normals):
    src_centroid = np.average(src_points, axis=0)
    tar_centroid = np.average(tar_points, axis=0)
    src_centered = (src_points - src_centroid) * 1e3
    tar_centered = (tar_points - tar_centroid) * 1e3
    src_stack = np.vstack([src_centered, src_normals])
    tar_stack = np.vstack([tar_centered, tar_normals])

    H = src_centered.T @ tar_centered
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    tar_centroid = np.mean(tar_points, axis=0)
    t = tar_centroid - np.dot(R, src_centroid)
    return (
        R,
        t,
    )


def process_candidate(
    candidate,
    src_points,
    tar_points,
    src_normals,
    tar_normals,
    tar_whole_points,
    tar_whole_normals,
    tar_tree,
):
    clique = np.array(candidate)
    src_pts = src_points[clique]
    tar_pts = tar_points[clique]
    src_norms = src_normals[clique]
    tar_norms = tar_normals[clique]

    try:
        R, t = svd_estimate_transform_normal(src_pts, tar_pts, src_norms, tar_norms)
    except np.linalg.LinAlgError:
        return None

    p_src = src_points @ R.T + t
    closest_tar_idx_of_copy_src = np.asarray(
        tar_tree.query(p_src, k=1, return_distance=False)
    )[:, 0]
    p_tar = tar_whole_points[closest_tar_idx_of_copy_src]
    n_tar = tar_whole_normals[closest_tar_idx_of_copy_src]

    A = np.hstack((np.cross(p_src, n_tar), n_tar))
    b = np.sum(n_tar * (p_tar - p_src), axis=1)

    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "R": R,
            "t": t,
            "residual": np.linalg.norm(b),
        }

    dR = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3])
    dt = x[3:]

    return {
        "R": dR @ R,
        "t": dR @ t + dt,
        "residual": np.linalg.norm(A @ x - b),
    }


def tacreg(
    src_pcd: o3d.geometry.PointCloud,
    tar_pcd: o3d.geometry.PointCloud,
    params: TacRegParam = TacRegParam(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform TAC-REG registration on source and target point clouds.

    Args:
        src_pcd (o3d.geometry.PointCloud): Source point cloud.
        tar_pcd (o3d.geometry.PointCloud): Target point cloud.
        correspondence (np.ndarray): Initial correspondences between source and target point clouds.
        params (TacRegParam): Parameters for TAC-REG.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation and translation matrices for registration.
    """
    src_iss, tar_iss, edges, _, tar_pcd_normals = fpfh_correspondence(
        src_pcd, tar_pcd, params=params
    )
    tar_whole_points = np.asarray(tar_pcd_normals.points)
    tar_whole_normals = np.asarray(tar_pcd_normals.normals)
    tar_tree = KDTree(tar_whole_points)

    src_points = np.asarray(src_iss.points)[edges[:, 0]]
    tar_points = np.asarray(tar_iss.points)[edges[:, 1]]
    src_edge_dists = np.linalg.norm(
        src_points[:, None, :] - src_points[None, :, :], axis=2
    )
    tar_edge_dists = np.linalg.norm(
        tar_points[:, None, :] - tar_points[None, :, :], axis=2
    )
    edges_dist_mat = np.abs(src_edge_dists - tar_edge_dists)

    src_normals = np.asarray(src_iss.normals)[edges[:, 0]]
    tar_normals = np.asarray(tar_iss.normals)[edges[:, 1]]
    src_edge_angles = np.arccos(
        np.clip(
            np.sum(src_normals[:, None, :] * src_normals[None, :, :], axis=2),
            -1.0,
            1.0,
        )
    )
    tar_edge_angles = np.arccos(
        np.clip(
            np.sum(tar_normals[:, None, :] * tar_normals[None, :, :], axis=2),
            -1.0,
            1.0,
        )
    )
    edges_angle_mat = np.abs(src_edge_angles - tar_edge_angles)

    dist_threshold = params.dist_threshold
    angle_threshold = params.angle_threshold
    valid_edges = (
        (edges_dist_mat < dist_threshold)
        & (edges_angle_mat < angle_threshold)
        & (~np.eye(edges.shape[0], dtype=bool))
    )
    if params.verbose:
        print(f"[Adj] Found {np.sum(valid_edges)} valid edges")
    for i in range(len(valid_edges[0])):
        for j in range(i + 1, len(valid_edges[0])):
            if valid_edges[i, j]:
                src_i, tar_i = edges[i]
                src_j, tar_j = edges[j]
                if src_i == src_j or tar_i == tar_j:
                    valid_edges[i, j] = False
                    valid_edges[j, i] = False
    if params.verbose:
        print(f"[Adj] Graph degree: {np.sum(valid_edges)} edges")

    G = nx.from_numpy_array(valid_edges)
    cliques = list(nx.find_cliques(G))
    candidates = sorted(
        [clique for clique in cliques if len(clique) > 2], key=len, reverse=True
    )[: params.candidate_size]
    if params.verbose:
        print("[Cand] Found {} candidates".format(len(candidates)))

    results = Parallel(n_jobs=params.parallel_jobs)(
        delayed(process_candidate)(
            candidate,
            src_points,
            tar_points,
            src_normals,
            tar_normals,
            tar_whole_points,
            tar_whole_normals,
            tar_tree,
        )
        for candidate in candidates
    )
    results = [res for res in results if res is not None]
    if not results:
        raise ValueError("No valid registration results found.")

    results = sorted(results, key=lambda x: x["residual"])
    best_result = results[0]

    return best_result["R"], best_result["t"]
