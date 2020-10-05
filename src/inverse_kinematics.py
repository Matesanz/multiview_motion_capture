import yaml
import numpy as np
import os
from scipy.optimize import least_squares
from typing import Optional
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from Quaternions import Quaternions
from util import descendants_mask
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pose_def import (KpsType, get_parent_index, KpsFormat, get_common_kps_idxs, get_common_kps_idxs_1, get_kps_index,
                      Pose, get_sides_joint_idxs, get_sides_joints, get_kps_order, get_joint_side, get_flip_joint)
from mv_math_util import triangulate_point_groups_from_multiple_views_linear

solver_verbose = 0
matplotlib.use('Qt5Agg')


def offsets_to_bone_dirs_bone_lens(offsets):
    bone_lens = np.linalg.norm(offsets, axis=-1)
    bdirs = offsets.copy()
    bdirs[1:, :] = bdirs[1:, :] / bone_lens[1:][:, np.newaxis]
    return bdirs, bone_lens


def bone_dir_bone_lens_to_offsets(bone_dirs, bone_lens):
    return bone_dirs * bone_lens[:, np.newaxis]


def plot_poses_3d(poses_3d: np.array, bones_idxs, target_pose, interval=50):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if target_pose is not None:
        for bone in bones_idxs:
            p0, p1 = target_pose[bone[0], :3], target_pose[bone[1], :3]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c='blue')

    bones_lines = [ax.plot([0, 0], [0, 0], [0, 0])[0] for _ in range(len(bones_idxs))]

    def _update_pose(frm_idx):
        for bone, line in zip(bones_idxs, bones_lines):
            p0, p1 = poses_3d[frm_idx, bone[0], :3], poses_3d[frm_idx, bone[1], :3]
            line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            line.set_3d_properties([p0[2], p1[2]])

    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z')

    # Creating the Animation object
    line_anim = animation.FuncAnimation(fig, _update_pose, len(poses_3d), interval=interval, blit=False)
    plt.show()


def plot_ik_result(pose_bones_colors):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    for (pose, bone_idxs, c) in pose_bones_colors:
        if bone_idxs is not None:
            for bone in bone_idxs:
                p0, p1 = pose[bone[0], :3], pose[bone[1], :3]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=c)
        else:
            ax.plot(pose[:, 0], pose[:, 1], pose[:, 2], '+r')
    plt.show()


def swap_y_z(poses):
    y = poses[..., 1].copy()
    z = poses[..., 2].copy()
    poses[..., 2] = y
    poses[..., 1] = z
    return poses


@dataclass
class PoseShapeParam:
    root: np.ndarray
    euler_angles: np.ndarray
    bone_lens: np.ndarray


@dataclass
class Skeleton:
    ref_joint_euler_angles: np.ndarray
    ref_bone_dirs: np.ndarray
    ref_side_bone_lens: np.ndarray  # left side plus mid bone lengths
    ref_side_to_full_bone_lens_map: List[int]
    n_joints: int
    joint_parents: np.ndarray
    kps_format: KpsFormat

    @property
    def skel_kps_idx_map(self):
        return get_kps_index(self.kps_format)

    @property
    def bone_idxs(self):
        bone_idxs = []
        for i, i_p in enumerate(self.joint_parents[1:]):
            bone_idxs.append((i + 1, i_p))
        return bone_idxs

    def to_full_bone_lens(self, side_blens):
        assert len(side_blens) == len(self.ref_side_bone_lens)
        return np.array([side_blens[idx] for idx in self.ref_side_to_full_bone_lens_map])


def load_skeleton():
    offsets = [
        [0, 0, 0],
        [0.15, 0, 0],
        [0, 0, -0.5],
        [0, 0, -0.5],
        [-0.15, 0, 0],
        [0, 0, -0.5],
        [0, 0, -0.5],
        [0, 0, 0.3],
        [0, 0, 0.3],
        [0.2, 0, 0],
        [0.3, 0, 0],
        [0.3, 0, 0],
        [-0.2, 0, 0],
        [-0.3, 0, 0],
        [-0.3, 0, 0],
        [0, -0.02, 0.15],
        [+0.07, 0.02, 0.1],
        [-0.07, 0.02, 0.1]
    ]

    kps_format = KpsFormat.BASIC_18

    skl_offsets, skl_parents = np.array(offsets), get_parent_index(kps_format)
    n_joints = len(skl_parents)
    bone_idxs = []
    for i, i_p in enumerate(skl_parents[1:]):
        bone_idxs.append((i + 1, i_p))

    skel_bdirs, skel_blens = offsets_to_bone_dirs_bone_lens(skl_offsets)

    kps_idx_map = get_kps_index(kps_format)
    ljoints, rjoints, mjoints = get_sides_joints(kps_format)
    l_m_joints = ljoints + mjoints
    l_m_skel_blens = [skel_blens[kps_idx_map[j_type]] for j_type in l_m_joints]
    # mapping from index of l_m_skel_blens to full bone length list.
    l_m_to_full_map = []
    for jnt_type, jnt_idx in kps_idx_map.items():
        jnt_side = get_joint_side(jnt_type)
        if jnt_side in ['left', 'mid']:
            l_m_to_full_map.append(l_m_joints.index(jnt_type))
        else:
            jnt_type = get_flip_joint(jnt_type)
            l_m_to_full_map.append(l_m_joints.index(jnt_type))

    skel = Skeleton(ref_joint_euler_angles=np.zeros((n_joints, 3)),
                    ref_bone_dirs=skel_bdirs,
                    ref_side_bone_lens=np.array(l_m_skel_blens),
                    ref_side_to_full_bone_lens_map=l_m_to_full_map,
                    joint_parents=skl_parents,
                    n_joints=len(skl_parents),
                    kps_format=kps_format)
    return skel


def foward_kinematics(skel: Skeleton, param: PoseShapeParam):
    root_loc = param.root
    rotations = Quaternions.from_euler(param.euler_angles)

    rot_mats = rotations.transforms()
    l_transforms = np.array([np.eye(4) for _ in range(skel.n_joints)])

    offsets = bone_dir_bone_lens_to_offsets(skel.ref_bone_dirs, skel.to_full_bone_lens(param.bone_lens))

    for j_i in range(skel.n_joints):
        l_transforms[j_i, :3, :3] = rot_mats[j_i]
        if j_i != 0:
            l_transforms[j_i, :3, 3] = offsets[j_i]
        else:
            if root_loc is not None:
                l_transforms[j_i, :3, 3] = root_loc

    g_transforms = l_transforms.copy()
    for j_i in range(1, skel.n_joints):
        g_transforms[j_i, :, :] = g_transforms[skel.joint_parents[j_i], :, :] @ l_transforms[j_i, :, :]

    g_pos = g_transforms[:, :, 3]
    g_pos = g_pos[:, :3] / g_pos[:, 3, np.newaxis]
    return g_pos, g_transforms


def solve_pose_reproj(skel: Skeleton,
                      obs_pose_2d: np.ndarray,
                      obs_kps_idxs: List[int],
                      cam_projs: List[np.ndarray],
                      skel_kps_idxs: List[int],
                      init_param: PoseShapeParam,
                      n_max_iter=5):
    obs_pose_2d = obs_pose_2d[:, obs_kps_idxs, :]
    init_locs, _ = foward_kinematics(skel, init_param)
    n_cams = len(cam_projs)

    def _decompose(_x: PoseShapeParam):
        return _x[:3], _x[3:].reshape((-1, 3))

    def _compose(p: PoseShapeParam):
        return np.concatenate([p.root.flatten(), p.euler_angles.flatten()])

    def _residual_step_joints_3d(_x):
        _root, _angles = _decompose(_x)
        _joint_locs, _ = foward_kinematics(skel, PoseShapeParam(_root, _angles, init_param.bone_lens))
        _joint_locs = _joint_locs[skel_kps_idxs, :]
        _n = len(_joint_locs)
        _joint_homo = np.concatenate([_joint_locs, np.ones((_n, 1), dtype=_joint_locs.dtype)], axis=-1).T

        _cam_kps_reproj = []
        for _vi in range(n_cams):
            _kps_proj = (cam_projs[_vi] @ _joint_homo)
            _kps_proj = (_kps_proj[:2] / (1e-5 + _kps_proj[2])).T
            _cam_kps_reproj.append(_kps_proj)
        _cam_kps_reproj = np.array(_cam_kps_reproj)
        _diffs = _cam_kps_reproj - obs_pose_2d[:, :, :2]
        _diffs = _diffs * obs_pose_2d[:, :, -1:]
        return _diffs.flatten()

    results = least_squares(_residual_step_joints_3d, _compose(init_param), verbose=solver_verbose, max_nfev=n_max_iter)
    root, angles = _decompose(results.x)
    return PoseShapeParam(root, angles, init_param.bone_lens)


def solve_pose_bone_lens_reproj(skel: Skeleton,
                                obs_pose_2d: np.ndarray,
                                obs_kps_idxs: List[int],
                                cam_projs: List[np.ndarray],
                                skel_kps_idxs: List[int],
                                init_param: PoseShapeParam,
                                n_max_iter=5):
    obs_pose_2d = obs_pose_2d[:, obs_kps_idxs, :]
    n_cams = len(cam_projs)
    n_joints = skel.n_joints

    def _decompose(_x):
        return _x[:3], _x[3:3 + n_joints * 3].reshape((-1, 3)), _x[3 + n_joints * 3:]

    def _compose(p: PoseShapeParam):
        return np.concatenate([p.root.flatten(), p.euler_angles.flatten(), p.bone_lens.flatten()])

    def _residual_root_angles_bone_lens(_x):
        _root, _angles, _blens = _decompose(_x)
        _joint_locs, _ = foward_kinematics(skel, PoseShapeParam(_root, _angles, _blens))
        _joint_locs = _joint_locs[skel_kps_idxs, :]
        _n = len(_joint_locs)
        _joint_homo = np.concatenate([_joint_locs, np.ones((_n, 1), dtype=_joint_locs.dtype)], axis=-1).T
        _cam_kps_reproj = []
        for _vi in range(n_cams):
            _kps_proj = (cam_projs[_vi] @ _joint_homo)
            _kps_proj = (_kps_proj[:2] / (1e-5 + _kps_proj[2])).T
            _cam_kps_reproj.append(_kps_proj)
        _cam_kps_reproj = np.array(_cam_kps_reproj)
        _diffs = _cam_kps_reproj - obs_pose_2d[:, :, :2]
        _diffs = _diffs * obs_pose_2d[:, :, -1:]
        return _diffs.flatten()

    results = least_squares(_residual_root_angles_bone_lens, _compose(init_param), verbose=solver_verbose,
                            max_nfev=n_max_iter)
    root, angles, blens = _decompose(results.x)
    return PoseShapeParam(root, angles, blens)


def solve_pose(skel: Skeleton,
               obs_pose_3d: np.ndarray,
               obs_kps_idxs: List[int],
               skel_kps_idxs: List[int],
               init_param: PoseShapeParam,
               n_max_iter=5) -> PoseShapeParam:
    init_locs, _ = foward_kinematics(skel, init_param)

    target_pose_3d_shared = obs_pose_3d[obs_kps_idxs, :]

    def _decompose(_x: PoseShapeParam):
        return _x[:3], _x[3:].reshape((-1, 3))

    def _compose(p: PoseShapeParam):
        return np.concatenate([p.root.flatten(), p.euler_angles.flatten()])

    def _residual_step_joints_3d(_x):
        _root, _angles = _decompose(_x)
        _joint_locs, _ = foward_kinematics(skel, PoseShapeParam(_root, _angles, init_param.bone_lens))
        _joint_locs = _joint_locs[skel_kps_idxs, :]
        _diffs = (_joint_locs - target_pose_3d_shared[:, :3])
        _diffs = _diffs * target_pose_3d_shared[:, -1:]
        return _diffs.flatten()

    results = least_squares(_residual_step_joints_3d, _compose(init_param), verbose=solver_verbose, max_nfev=n_max_iter)
    root, angles = _decompose(results.x)
    return PoseShapeParam(root, angles, init_param.bone_lens)


def solve_pose_bone_lens(skel: Skeleton,
                         obs_pose_3d: np.ndarray,
                         obs_kps_idxs: List[int],
                         skel_kps_idxs: List[int],
                         init_param: PoseShapeParam,
                         n_max_iter=5):
    target_pose_3d_shared = obs_pose_3d[obs_kps_idxs, :]
    n_joints = skel.n_joints

    def _decompose(_x):
        return _x[:3], _x[3:3 + n_joints * 3].reshape((-1, 3)), _x[3 + n_joints * 3:]

    def _compose(p: PoseShapeParam):
        return np.concatenate([p.root.flatten(), p.euler_angles.flatten(), p.bone_lens.flatten()])

    def _residual_root_angles_bone_lens(_x):
        _root, _angles, _blens = _decompose(_x)
        _joint_locs, _ = foward_kinematics(skel,
                                           PoseShapeParam(_root, _angles, _blens))
        _joint_locs = _joint_locs[skel_kps_idxs, :]
        _diffs = (_joint_locs - target_pose_3d_shared[:, :3])
        _diffs = _diffs * target_pose_3d_shared[:, -1:]
        return _diffs.flatten()

    results = least_squares(_residual_root_angles_bone_lens, _compose(init_param), verbose=solver_verbose,
                            max_nfev=n_max_iter)
    root, angles, blens = _decompose(results.x)
    return PoseShapeParam(root, angles, blens)


def guess_mid_spine(pose_2d: np.ndarray, kps_idx_map):
    if KpsType.Spine not in kps_idx_map:
        mid_shoulder = 0.5 * (pose_2d[kps_idx_map[KpsType.L_Shoulder], :] + pose_2d[kps_idx_map[KpsType.R_Shoulder], :])
        midhip = 0.5 * (pose_2d[kps_idx_map[KpsType.L_Hip], :] + pose_2d[kps_idx_map[KpsType.R_Hip], :])
        spine = 0.5 * (mid_shoulder + midhip)
        score = pose_2d[kps_idx_map[KpsType.L_Shoulder], -1] * pose_2d[kps_idx_map[KpsType.R_Shoulder], -1]
        score *= pose_2d[kps_idx_map[KpsType.L_Hip], -1] * pose_2d[kps_idx_map[KpsType.R_Hip], -1]
        return np.array([spine[0], spine[1], score])
    else:
        return None


class PoseSolver:
    def __init__(self,
                 skeleton: Skeleton,
                 init_pose: Optional[PoseShapeParam],
                 cam_poses_2d: List[np.ndarray],
                 cam_projs: List[np.ndarray],
                 obs_kps_format: KpsFormat):
        self.skel = skeleton
        self.n_joints = self.skel.n_joints
        self.init_pose = init_pose
        self.cam_poses_2d = cam_poses_2d
        self.cam_projs = cam_projs
        self.obs_kps_format = obs_kps_format
        self.obs_kps_idx_map = get_kps_index(self.obs_kps_format)
        self.skel_kps_format = self.skel.kps_format
        self._hack_add_midspine()
        self.skel_kps_idxs, self.obs_kps_idxs = get_common_kps_idxs_1(get_kps_index(self.skel_kps_format),
                                                                      self.obs_kps_idx_map)

    def _hack_add_midspine(self):
        new_cam_poses = []
        n_old_kps = len(self.obs_kps_idx_map)
        for pose in self.cam_poses_2d:
            mid_spine = guess_mid_spine(pose, self.obs_kps_idx_map)
            pose_new = np.concatenate([pose, mid_spine.reshape((-1, 3))], axis=0)
            new_cam_poses.append(pose_new)
        self.cam_poses_2d = new_cam_poses
        self.obs_kps_idx_map[KpsType.Spine] = n_old_kps

    def solve(self) -> Tuple[PoseShapeParam, Pose]:
        # debug
        # return (PoseShapeParam(root=np.zeros((3,)),
        #                        euler_angles=np.zeros((17, 3)), bone_lens=np.zeros((17, 3))),
        #         Pose(keypoints=obs_pose_3d[:, :3],
        #              keypoints_score=obs_pose_3d[:, -1],
        #              box=None,
        #              pose_type=KpsFormat.COCO))

        if self.init_pose is None:
            obs_pose_3d = triangulate_point_groups_from_multiple_views_linear(self.cam_projs,
                                                                              self.cam_poses_2d, 0.01, True)
            init_root = 0.5 * (obs_pose_3d[self.obs_kps_idx_map[KpsType.L_Hip], :3] +
                               obs_pose_3d[self.obs_kps_idx_map[KpsType.R_Hip], :3])
            init_blens = self.skel.ref_side_bone_lens.copy()
            init_angles = np.zeros((self.n_joints, 3), dtype=init_root.dtype)
            init_param = PoseShapeParam(init_root, init_angles, init_blens)
            n_max_iter = 50
        else:
            init_param = self.init_pose
            n_max_iter = 5

        use_only_reproj = True
        if use_only_reproj:
            param_1 = solve_pose_reproj(self.skel, np.array(self.cam_poses_2d), self.obs_kps_idxs, self.cam_projs,
                                        self.skel_kps_idxs, init_param, n_max_iter)
            param_2 = solve_pose_bone_lens_reproj(self.skel, np.array(self.cam_poses_2d), self.obs_kps_idxs,
                                                  self.cam_projs, self.skel_kps_idxs, param_1, n_max_iter)
        else:
            obs_pose_3d = triangulate_point_groups_from_multiple_views_linear(self.cam_projs,
                                                                              self.cam_poses_2d, 0.01, True)
            param_1 = solve_pose(self.skel, obs_pose_3d, self.obs_kps_idxs, self.skel_kps_idxs, init_param, n_max_iter)
            param_2 = solve_pose_bone_lens(self.skel, obs_pose_3d, self.obs_kps_idxs, self.skel_kps_idxs, param_1,
                                           n_max_iter)

        pred_locs_2, _ = foward_kinematics(self.skel, param_2)

        debug = False
        if debug:
            pred_locs_1, _ = foward_kinematics(self.skel, param_1)
            init_locs, _ = foward_kinematics(self.skel, init_param)
            obs_pose_3d = triangulate_point_groups_from_multiple_views_linear(self.cam_projs,
                                                                              self.cam_poses_2d, 0.01, True)
            pose_bones_colors = [
                (init_locs, self.skel.bone_idxs, 'red'),
                (pred_locs_1, self.skel.bone_idxs, 'green'),
                (pred_locs_2, self.skel.bone_idxs, 'blue'),
                (obs_pose_3d, None, 'magnenta')]
            plot_ik_result(pose_bones_colors)

        return param_2, Pose(keypoints=pred_locs_2,
                             keypoints_score=np.ones((len(pred_locs_2), 1)),
                             box=None,
                             pose_type=KpsFormat.BASIC_18)


def run_test_ik(target_pose_3d: np.ndarray, cam_poses_2d: List[np.ndarray], cam_projs: List[np.ndarray]):
    obs_kps_format = KpsFormat.COCO
    skel_kps_format = KpsFormat.BASIC_18
    obs_kps_idx_map = get_kps_index(obs_kps_format)

    skel_kps_idxs, obs_kps_idxs = get_common_kps_idxs(skel_kps_format, obs_kps_format)
    cam_poses_2d_match = [pose[obs_kps_idxs, :] for pose in cam_poses_2d]
    target_pose_3d_match = target_pose_3d[obs_kps_idxs, :]

    skl_offsets, skl_parents = load_skeleton()
    skl_descendants = descendants_mask(skl_parents)
    n_joints = len(skl_parents)
    bone_idxs = []
    for i, i_p in enumerate(skl_parents[1:]):
        bone_idxs.append((i + 1, i_p))

    skel_bdirs, skel_blens = offsets_to_bone_dirs_bone_lens(skl_offsets)
    skel = Skeleton(ref_joint_euler_angles=np.zeros((n_joints, 3)),
                    ref_bone_dirs=skel_bdirs,
                    ref_bone_lens=skel_blens,
                    joint_parents=skl_parents,
                    n_joints=len(skl_parents),
                    kps_format=KpsFormat.BASIC_18)
    solver = PoseSolver(skel, init_pose=None, cam_poses_2d=cam_poses_2d, cam_projs=cam_projs,
                        obs_kps_format=KpsFormat.COCO)
    solver.solve()

    # def _residual_step_2(_x):
    #     _root_pos = _x[:3]
    #     _joint_quars = Quaternions.from_euler(_x[3:].reshape((-1, 3)))
    #     _joint_locs, _ = model.forward(_joint_quars, _root_pos)
    #     _joint_locs = _joint_locs[skel_kps_idxs, :]
    #
    #     _joint_homos = np.concatenate([_joint_locs, np.ones((_joint_locs.shape[0], 1))], axis=-1).T
    #     _diff_reprojs = []
    #     for _vi in range(n_cams):
    #         _proj = cam_projs[_vi] @ _joint_homos
    #         _proj = (_proj[:2] / _proj[2]).T
    #         _d = np.linalg.norm(_proj - cam_poses_2d_match[_vi][:, :2], axis=-1)
    #         _d = _d * cam_poses_2d_match[_vi][:, -1]
    #         _diff_reprojs.append(_d)
    #     _diff_reprojs = np.array(_diff_reprojs).flatten()
    #     return _diff_reprojs
