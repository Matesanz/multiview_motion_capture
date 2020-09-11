import yaml
import numpy as np
import os
from scipy.optimize import least_squares
from typing import Optional
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from kinematics.Quaternions import Quaternions
from kinematics.util import descendants_mask
from typing import List
from pose_def import KpsType, get_parent_index, KpsFormat, get_common_kps_idxs, get_pose_bones_index, get_kps_index

matplotlib.use('Qt5Agg')


class Skeleton:
    def __init__(self):
        filename = '/media/F/thesis/libs/deep-motion-editing/style_transfer/global_info/skeleton_CMU.yml'
        with open(filename, "r") as file:
            skel = yaml.load(file, Loader=yaml.Loader)
        self.bvh_name = os.path.join(os.path.dirname(filename), skel['BVH'])
        self.offset = np.array(skel['offsets'])
        self.n_joints = len(self.offset)
        self.topology = np.array(skel['parents'])
        self.chosen_joints = np.array(skel['chosen_joints'])
        self.chosen_parents = np.array(skel['chosen_parents'])
        self.hips, self.sdrs = skel['hips'], skel['shoulders']
        self.head = skel['head']


class ForwardKinematics:
    def __init__(self, offsets, parents):
        self.offset = offsets
        self.parents = parents
        self.n_joints = len(self.offset)

    def forward(self, rotations: Quaternions, root_loc: Optional[np.ndarray]):
        rot_mats = rotations.transforms()
        l_transforms = np.array([np.eye(4) for _ in range(self.n_joints)])
        for j_i in range(self.n_joints):
            l_transforms[j_i, :3, :3] = rot_mats[j_i]
            if j_i != 0:
                l_transforms[j_i, :3, 3] = self.offset[j_i]
            else:
                if root_loc is not None:
                    l_transforms[j_i, :3, 3] = root_loc

        g_transforms = l_transforms.copy()
        for j_i in range(1, self.n_joints):
            g_transforms[j_i, :, :] = g_transforms[self.parents[j_i], :, :] @ l_transforms[j_i, :, :]

        g_pos = g_transforms[:, :, 3]
        g_pos = g_pos[:, :3] / g_pos[:, 3, np.newaxis]
        return g_pos, g_transforms


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


def plot_ik_result(init_pose, pred_pose, target_pose, bone_idxs, target_bone_idxs):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    n_joints = len(init_pose)
    # for j_idx in range(n_joints):
    #     ax.scatter(init_pose[j_idx, 0], init_pose[j_idx, 1], init_pose[j_idx, 2])
    #     ax.text(init_pose[j_idx, 0], init_pose[j_idx, 1], init_pose[j_idx, 2], f'{j_idx}', size=10, zorder=1, color='k')

    for bone in bone_idxs:
        p0, p1 = init_pose[bone[0], :3], init_pose[bone[1], :3]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='red')

    for bone in bone_idxs:
        p0, p1 = pred_pose[bone[0], :3], pred_pose[bone[1], :3]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='magenta')

    if target_bone_idxs is not None:
        for bone in bone_idxs:
            p0, p1 = target_pose[bone[0], :3], target_pose[bone[1], :3]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='blue')
    else:
        ax.plot(target_pose[:, 0], target_pose[:, 1], target_pose[:, 2], '+r')

    plt.show()


def load_skeleton():
    from kinematics import bvh
    # bvh_path = '/media/F/thesis/libs/deep-motion-editing/style_transfer/data/xia_test/depressed_13_000.bvh'
    bvh_path = '/media/F/thesis/multiview_motion_capture/src/simple.bvh'
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
        [0, 0, 0.3],
        [0.07, 0, 0.1],
        [-0.07, 0, 0.1]
    ]

    return np.array(offsets), get_parent_index(KpsFormat.BASIC_18)


def swap_y_z(poses):
    y = poses[..., 1].copy()
    z = poses[..., 2].copy()
    poses[..., 2] = y
    poses[..., 1] = z
    return poses


def triangulate_optimize(target_pose_3d: np.ndarray, cam_poses_2d: List[np.ndarray], cam_projs: List[np.ndarray]):
    in_kps_format = KpsFormat.COCO

    bone_idxs = get_pose_bones_index(in_kps_format)
    n_cams = len(cam_projs)

    def _residual(_x):
        # return _diffs.flatten()
        _joint_locs = _x.reshape((-1, 3))
        _joint_homos = np.concatenate([_joint_locs, np.ones((_joint_locs.shape[0], 1))], axis=-1).T
        _diff_reprojs = []
        for _vi in range(n_cams):
            _proj = cam_projs[_vi] @ _joint_homos
            _proj = (_proj[:2] / _proj[2]).T
            # _d = (_proj - cam_poses_2d[_vi][:, :2]) * cam_poses_2d[_vi][:, -1:]
            _d = np.linalg.norm(_proj - cam_poses_2d[_vi][:, :2], axis=-1)
            _d = _d * cam_poses_2d[_vi][:, -1]
            _diff_reprojs.append(_d)
        _diff_reprojs = np.array(_diff_reprojs).flatten()

        return _diff_reprojs

    params = np.zeros((len(target_pose_3d), 3)).flatten()

    results = least_squares(_residual, params, verbose=2, ftol=None, max_nfev=100)
    params_1 = results.x

    pred_locs = params_1.reshape((-1, 3))

    # plot_ik_result(pred_locs, pred_locs, target_pose=target_pose_3d, bone_idxs=bone_idxs, target_bone_idxs=None)

    return pred_locs


def run_test_ik(target_pose_3d: np.ndarray, cam_poses_2d: List[np.ndarray], cam_projs: List[np.ndarray]):
    target_pose_3d = triangulate_optimize(target_pose_3d, cam_poses_2d, cam_projs)

    in_kps_format = KpsFormat.COCO
    my_kps_format = KpsFormat.BASIC_18
    in_kps_idx_map = get_kps_index(in_kps_format)

    my_kps_idxs, in_kps_idxs = get_common_kps_idxs(my_kps_format, in_kps_format)
    cam_poses_2d_match = [pose[in_kps_idxs, :] for pose in cam_poses_2d]
    target_pose_3d_match = target_pose_3d[in_kps_idxs, :]

    skl_offsets, skl_parents = load_skeleton()
    skl_descendants = descendants_mask(skl_parents)
    n_joints = len(skl_parents)
    bone_idxs = []
    for i, i_p in enumerate(skl_parents[1:]):
        bone_idxs.append((i + 1, i_p))

    model = ForwardKinematics(skl_offsets, skl_parents)
    n_cams = len(cam_projs)

    def _residual_step_1(_x):
        _root_pos = _x[:3]
        _joint_quars = Quaternions.from_euler(_x[3:].reshape((-1, 3)))
        _joint_locs, _ = model.forward(_joint_quars, _root_pos)
        _joint_locs = _joint_locs[my_kps_idxs, :]
        _diffs = (_joint_locs - target_pose_3d_match).flatten()
        return _diffs

    def _residual_step_2(_x):
        _root_pos = _x[:3]
        _joint_quars = Quaternions.from_euler(_x[3:].reshape((-1, 3)))
        _joint_locs, _ = model.forward(_joint_quars, _root_pos)
        _joint_locs = _joint_locs[my_kps_idxs, :]

        _joint_homos = np.concatenate([_joint_locs, np.ones((_joint_locs.shape[0], 1))], axis=-1).T
        _diff_reprojs = []
        for _vi in range(n_cams):
            _proj = cam_projs[_vi] @ _joint_homos
            _proj = (_proj[:2] / _proj[2]).T
            _d = np.linalg.norm(_proj - cam_poses_2d_match[_vi][:, :2], axis=-1)
            _d = _d * cam_poses_2d_match[_vi][:, -1]
            _diff_reprojs.append(_d)
        _diff_reprojs = np.array(_diff_reprojs).flatten()
        return _diff_reprojs

    params = np.zeros((3 + np.prod(skl_offsets.shape)))
    init_root = 0.5 * (
            target_pose_3d[in_kps_idx_map[KpsType.L_Hip], :3] + target_pose_3d[in_kps_idx_map[KpsType.R_Hip], :3])
    params[:3] = init_root

    init_root = params[:3]
    init_quars = Quaternions.from_euler(params[3:].reshape((-1, 3)))
    init_locs, _ = model.forward(init_quars, init_root)

    results = least_squares(_residual_step_1, params, verbose=2, max_nfev=100)
    params_1 = results.x
    pred_root_1 = params_1[:3]
    pred_quars_1 = Quaternions.from_euler(params_1[3:].reshape((-1, 3)))
    pred_locs_1, _ = model.forward(pred_quars_1, pred_root_1)

    results = least_squares(_residual_step_2, params_1, verbose=2, max_nfev=100)
    params_2 = results.x
    pred_root_2 = params_2[:3]
    pred_quars_2 = Quaternions.from_euler(params_2[3:].reshape((-1, 3)))
    pred_locs_2, _ = model.forward(pred_quars_2, pred_root_2)

    plot_ik_result(pred_locs_1, pred_locs_2, target_pose=target_pose_3d, bone_idxs=bone_idxs, target_bone_idxs=None)
