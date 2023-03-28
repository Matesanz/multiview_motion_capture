import os.path as osp
import sys
import time
import torch
import numpy as np
import cv2
from pose_def import Pose
from typing import List
from scipy.sparse.linalg import eigs
from mv_math_util import Calib, calc_pairwise_f_mats, geometry_affinity
from scipy.optimize import linear_sum_assignment
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable


import numpy as np
from typing import List

def transform_closure(x_bin: np.ndarray) -> np.ndarray:
    """
    Convert binary relation matrix to permutation matrix.
    
    Args:
        x_bin (np.ndarray): Binarized relation matrix (numpy array) obtained by applying a threshold.
    
    Returns:
        np.ndarray: Transformed permutation matrix.
    """
    def transitive_closure(matrix: np.ndarray) -> np.ndarray:
        """Compute the transitive closure of the input matrix."""
        N = matrix.shape[0]
        temp = np.zeros_like(matrix)
        
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    temp[i, j] = matrix[i, j] or (matrix[i, k] and matrix[k, j])
        
        return temp

    def create_permutation_matrix(temp_matrix: np.ndarray, size: int) -> np.ndarray:
        """Create a permutation matrix from the transitive closure matrix."""
        visited = np.zeros(size)
        perm_matrix = np.zeros_like(temp_matrix)

        for i, row in enumerate(temp_matrix):
            if visited[i]:
                continue
            for j, is_relative in enumerate(row):
                if is_relative:
                    visited[j] = 1
                    perm_matrix[j, i] = 1
                    
        return perm_matrix

    transitive_closure_matrix = transitive_closure(x_bin)
    permutation_matrix = create_permutation_matrix(transitive_closure_matrix, x_bin.shape[0])

    return permutation_matrix


def match_als(W: np.ndarray, dimGroup: List[int], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the following optimization problem:
    min - <W,X> + alpha||X||_* + beta||X||_1, s.t. X \in C

    The problem is rewritten as
    <beta-W,AB^T> + alpha/2||A||^2 + alpha/2||B||^2
    s.t AB^T=Z, Z\in\Omega

    And returns 
    X: a sparse binary matrix indicating correspondences
    A: AA^T = X;
    info: other info.


    Other options are
    - maxRank: the restricted rank of X* (select it as large as possible)
    - alpha: the weight of nuclear norm
    - beta: the weight of l1 norm
    - pSelect: propotion of selected points, i.e., m'/m in section 5.4 in the paper
    - tol: tolerance of convergence
    - maxIter: maximal iteration
    - verbose: display info or not
    - eigenvalues: output eigenvalues or not

    Args:
        W (np.ndarray): Sparse input matrix storing scores of pairwise matches.
        dimGroup (List[int]): A list storing the number of points on each object.
        kwargs (Dict): Optional parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the 
            sparse binary matrix indicating correspondences and 
            the transformed permutation matrix.
    """
    # Optional parameters
    alpha = kwargs.get("alpha", 50)
    beta = kwargs.get("beta", 0.1)
    n_max_pp = np.diff(dimGroup)
    maxRank = kwargs.get("maxRank", max(n_max_pp) * 2)
    pSelect = kwargs.get("pSelect", 1)
    tol = kwargs.get("tol", 1e-4)
    maxIter = kwargs.get("maxIter", 1000)
    verbose = kwargs.get("verbose", False)
    eigenvalues = kwargs.get("eigenvalues", False)
    
    W = 0.5 * (W + W.T)
    X = W.copy()
    Z = W.copy()
    Y = np.zeros_like(W)
    mu = 64
    n = X.shape[0]
    maxRank = min(n, maxRank)

    A = np.random.RandomState(0).rand(n, maxRank)

    iter_cnt = 0
    t0 = time.time()

    for iter_idx in range(maxIter):
        X0 = X.copy()
        X = Z - (Y - W + beta) / mu
        B = (np.linalg.inv(A.T @ A + alpha / mu * np.eye(maxRank)) @ (A.T @ X)).T
        A = (np.linalg.inv(B.T @ B + alpha / mu * np.eye(maxRank)) @ (B.T @ X.T)).T
        X = A @ B.T

        Z = X + Y / mu
        # Enforce the self-matching to be null
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            Z[ind1:ind2, ind1:ind2] = 0

        if pSelect == 1:
            Z[np.arange(n), np.arange(n)] = 1

        Z[Z < 0] = 0
        Z[Z > 1] = 1

        Y = Y + mu * (X - Z)

        # Test for convergence
        pRes = np.linalg.norm(X - Z) / n
        dRes = mu * np.linalg.norm(X - X0) / n

        if pRes < tol and dRes < tol:
            iter_cnt = iter_idx
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = 0.5 * (X + X.T)
    X_bin = X > 0.5

    total_time = time.time() - t0

    match_mat = transform_closure(X_bin)

    return match_mat, X_bin


def match_multiview_poses(cam_poses: List[List[Pose]], calibs: List[Calib]):
    points_set = []
    dimsGroup = [0]
    cnt = 0
    for poses in cam_poses:
        cnt += len(poses)
        dimsGroup.append(cnt)
        for p in poses:
            points_set.append(p.keypoints)

    points_set = np.array(points_set)
    pairwise_f_mats = calc_pairwise_f_mats(calibs)
    s_mat = geometry_affinity(points_set, pairwise_f_mats, dimsGroup)
    # match_mat = matchSVT(torch.from_numpy(s_mat), dimsGroup)
    match_mat = match_als(s_mat, dimsGroup)

    bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
    bin_match = bin_match.reshape(s_mat.shape[0], -1)
    matched_list = [[] for i in range(bin_match.shape[1])]
    for sub_imgid, row in enumerate(bin_match):
        if row.sum() != 0:
            pid = row.numpy().argmax()
            matched_list[pid].append(sub_imgid)

    outputs = []
    for matches in matched_list:
        cam_p_idxs = []
        for idx in matches:
            cam_offset = 0
            cam_idx = 0
            for cur_cam_idx, offset in enumerate(dimsGroup):
                if offset <= idx:
                    cam_offset = offset
                    cam_idx = cur_cam_idx
                else:
                    break

            p_idx = idx - cam_offset
            cam_p_idxs.append((cam_idx, p_idx))

        if cam_p_idxs:
            outputs.append(cam_p_idxs)

    return outputs
