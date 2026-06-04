"""
Combined ROS1 bag reader + OLS dynamic identification analysis.
This script reads one or more ROS1 bags containing dVRK trajectory data, processes
The script was tested under conda python 3.8.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# rosbags: pure Python, no ROS installation needed (pip install rosbags)
import glob
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

# Dynamic identification imports (from main_psm_si_all.ipynb cell 43)
from dvrk_dynamic_identification.utils import load_data
from dvrk_dynamic_identification.identification.data_processing import (
    diff_and_filt_data, plot_meas_pred_tau, gen_regressor
)

# Usage example (run from project root, --train/--test take bag directories):
#python3 excute_run/Traj_read_ros1_analyze_multi.py --train data/ROS1_BIGFrame/train.bag --test data/ROS1_BIGFrame/test.bag --robot PSM2 --hz 200 --model-folder data/psm_si/model/ --model-name psm_si_optimal_0_Moter_inertia

# Bag readings
# adapted from Traj_read_analyze_multi.py to use rosbag (ROS1) instead of mcap (ROS2)

_typestore = get_typestore(Stores.ROS1_NOETIC)


def _read_topic(bag_path, topic):
    """Read all messages on a topic from a ROS1 bag file. Returns (timestamps, msgs)."""
    timestamps, msgs = [], []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            return timestamps, msgs
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            timestamps.append(timestamp * 1e-9)  # nanoseconds to seconds
            msgs.append(_typestore.deserialize_ros1(rawdata, connection.msgtype))
    return timestamps, msgs


def read_single_bag(bag_path, robot):
    """Read one bag, merge jaw as joint-7. Returns (t, q, dq, tau).
    Mirrors read_ros2_bags.py :: single_datapoint_processing."""
    joint_timestamps, joint_position, joint_velocity, joint_effort = [], [], [], []
    jaw_timestamps, jaw = [], []

    # --- measured_js ---
    ts_list, msgs = _read_topic(bag_path, f'/{robot}/measured_js')
    expected_dof = None
    for ts, msg in zip(ts_list, msgs):
        if len(msg.position) == 0:
            continue
        if expected_dof is None:
            expected_dof = len(msg.position)
        if len(msg.position) != expected_dof:
            continue
        joint_timestamps.append(ts)
        joint_position.append(list(msg.position))
        joint_velocity.append(list(msg.velocity))
        joint_effort.append(list(msg.effort))

    # --- jaw measured_js ---
    ts_list, msgs = _read_topic(bag_path, f'/{robot}/jaw/measured_js')
    for ts, msg in zip(ts_list, msgs):
        if len(msg.position) == 0:
            continue
        jaw_timestamps.append(ts)
        jaw.append([msg.position[0], msg.velocity[0], msg.effort[0]])

    print("Processed measured_js: count: {}".format(len(joint_timestamps)))
    print("Processed jaw measured_js: count: {}".format(len(jaw_timestamps)))

    # --- build arrays (mirroring read_ros2_bags.py) ---
    start_time = joint_timestamps[0]
    joint_timestamps = np.array(joint_timestamps) - start_time

    # joints shape: (N, 1 + dof + dof + dof)  — first col is time
    joints = np.column_stack((joint_timestamps, joint_position, joint_velocity, joint_effort))

    if len(jaw) > 0:
        jaw_timestamps = np.array(jaw_timestamps) - start_time
        jaw_mat = np.column_stack((jaw_timestamps, jaw))   # (M, 4): t, q, dq, tau
        n = min(len(joints), len(jaw_mat))
        joints  = joints[:n]
        jaw_mat = jaw_mat[:n]
        t       = joints[:, 0]
        q_raw   = np.column_stack([joints[:, 1:1+expected_dof],              jaw_mat[:, 1:2]])
        dq_raw  = np.column_stack([joints[:, 1+expected_dof:1+2*expected_dof], jaw_mat[:, 2:3]])
        tau_raw = np.column_stack([joints[:, 1+2*expected_dof:],              jaw_mat[:, 3:4]])
    else:
        t       = joints[:, 0]
        q_raw   = joints[:, 1:1+expected_dof]
        dq_raw  = joints[:, 1+expected_dof:1+2*expected_dof]
        tau_raw = joints[:, 1+2*expected_dof:]

    return t, q_raw, dq_raw, tau_raw





def main():
    parser = argparse.ArgumentParser()
    # bag reading args
    parser.add_argument('--train',  required=True, type=str, help='Train bag file (.bag)')
    parser.add_argument('--test',   required=True, type=str, help='Test bag file (.bag)')
    parser.add_argument('--robot',  default='PSM2', type=str)
    # analysis args (from main_psm_si_all.ipynb)
    parser.add_argument('--hz',          default=200,   type=int,   help='Sampling rate (Hz)')
    parser.add_argument('--model-folder', default='data/psm_si/model/', type=str)
    parser.add_argument('--model-name',   default='psm_si',              type=str)
    parser.add_argument('--base-freq',    default=0.18, type=float)
    parser.add_argument('--fourier-order', default=6,   type=int)
    parser.add_argument('--fc-mult',      default=5.0,  type=float,
                        help='fc = fc_mult * base_freq * fourier_order  (main_psm_si_all)')
    parser.add_argument('--cut-num',        default=98,  type=int, help='Train cut_num')
    parser.add_argument('--post-cut',       default=10,  type=int, help='Train post_cut')
    parser.add_argument('--cut-num-test',   default=97,  type=int, help='Test cut_num')
    parser.add_argument('--post-cut-test',  default=80,  type=int, help='Test post_cut')
    parser.add_argument('--output', default=None, type=str,
                        help='Folder to save plots and NRMS results (optional)')
    args = parser.parse_args()

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # --- Read bags
    print("\n=== Reading train data ===")
    print("Reading " + args.train)
    _, q_raw_train, dq_raw_train, tau_raw_train = read_single_bag(args.train, args.robot)
    print("\n=== Reading test data ===")
    print("Reading " + args.test)
    _, q_raw_test,  dq_raw_test,  tau_raw_test  = read_single_bag(args.test,  args.robot)

    # --- Load robot model (from main_psm_si_all.ipynb cell 30) ---
    robot_model = load_data(args.model_folder, args.model_name)
    dof = robot_model.dof

    # --- fc (from main_psm_si_all.ipynb cell 54) ---
    h  = 1.0 / args.hz
    fc = np.array([args.fc_mult]) * args.base_freq * args.fourier_order
    print("\nfc =", fc)

    # --- Generate t from sample index (from data_processing.py :: load_trajectory_data) ---
    t_train = np.arange(len(q_raw_train), dtype=float) / args.hz
    t_test  = np.arange(len(q_raw_test),  dtype=float) / args.hz

    # --- Diff and filter train (from main_psm_si_all.ipynb cell 57) ---
    t_cut_train, q_f_train, dq_f_train, ddq_f_train, tau_f_train, _, _ = \
        diff_and_filt_data(dof, h, t_train, q_raw_train, dq_raw_train, tau_raw_train,
                           fc, fc, fc, fc, cut_num=args.cut_num, post_cut=args.post_cut)
    print('train: raw=%d  after cut=%d' % (len(tau_raw_train), len(tau_f_train)))

    # --- Diff and filter test (from main_psm_si_all.ipynb cell 59) ---
    t_cut_test, q_f_test, dq_f_test, ddq_f_test, tau_f_test, _, _ = \
        diff_and_filt_data(dof, h, t_test, q_raw_test, dq_raw_test, tau_raw_test,
                           fc, fc, fc, fc, cut_num=args.cut_num_test, post_cut=args.post_cut_test)
    print('test:  raw=%d  after cut=%d' % (len(tau_raw_test),  len(tau_f_test)))

    # --- OLS (from main_psm_si_all.ipynb cells 63, 65) ---
    base_param_num = robot_model.base_num
    H_b_func = robot_model.H_b_func
    W_b_train, tau_s_train = gen_regressor(base_param_num, H_b_func,
                                           q_f_train, dq_f_train, ddq_f_train, tau_f_train)
    xb_ols = np.linalg.lstsq(W_b_train, tau_s_train, rcond=None)[0]

    # Predicted torque train
    tau_p_train = np.zeros(tau_f_train.shape)
    tau_ps_train = W_b_train.dot(xb_ols)
    for i in range(dof):
        tau_p_train[:, i] = tau_ps_train[i::dof]

    # --- Plot train (from main_psm_si_all.ipynb cell 75) ---
    plot_meas_pred_tau(t_cut_train, tau_f_train, tau_p_train,
                       robot_model.coordinates_joint_type, robot_model.coordinates)
    if args.output:
        plt.savefig(os.path.join(args.output, 'plot_train.png'), dpi=150, bbox_inches='tight')
        print('Saved plot_train.png')

    # --- Test regressor + predicted torque (from main_psm_si_all.ipynb cells 77-78) ---
    W_b_test, tau_s_test = gen_regressor(base_param_num, H_b_func,
                                         q_f_test, dq_f_test, ddq_f_test, tau_f_test)
    tau_p_test = np.zeros(tau_f_test.shape)
    tau_ps_test = W_b_test.dot(xb_ols)
    for i in range(dof):
        tau_p_test[:, i] = tau_ps_test[i::dof]

    # --- Plot test (from main_psm_si_all.ipynb cell 78) ---
    plot_meas_pred_tau(t_cut_test, tau_f_test, tau_p_test,
                       robot_model.coordinates_joint_type, robot_model.coordinates)
    # if args.output:
    #     plt.savefig(os.path.join(args.output, 'plot_test.png'), dpi=150, bbox_inches='tight')
    #     print('Saved plot_test.png')

    # --- NRMS error per joint (from main_psm_si_all.ipynb) ---
    nrms_train = (np.linalg.norm(tau_f_train - tau_p_train, axis=0)
                  / np.linalg.norm(tau_f_train, axis=0))
    nrms_test  = (np.linalg.norm(tau_f_test  - tau_p_test,  axis=0)
                  / np.linalg.norm(tau_f_test,  axis=0))
    print('\nNRMS per joint (train):', nrms_train)
    print('NRMS per joint (test): ', nrms_test)
    if args.output:
        nrms_path = os.path.join(args.output, 'nrms.csv')
        header = 'joint,' + ','.join(f'j{i+1}' for i in range(dof))
        np.savetxt(nrms_path,
                   np.vstack([nrms_train, nrms_test]),
                   delimiter=',', header=header,
                   comments='', fmt='%.6f')
        print('Saved nrms.csv')


if __name__ == '__main__':
    main()
