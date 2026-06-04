import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from dvrk_dynamic_identification.utils import load_data
from dvrk_dynamic_identification.identification.data_processing import (
    diff_and_filt_data, plot_meas_pred_tau, gen_regressor
)

# ============================================================
# INPUT — edit these
# ============================================================
train_bag      = 'data/ROS1_BIGFrame/traj_opt_PSM1_01_trajectory_output_29.bag'
test_bag       = 'data/ROS1_BIGFrame/test_psm1_openH.bag'
robot          = 'PSM1'
hz             = 200
model_folder   = 'data/psm_si/model/'
model_name     = 'psm_si_45_45'
traj_folder    = 'data/psm_si/optimal_trajectory/'
traj_name      = 'one'
xb_ols_path    = None   # path to pre-computed xb_ols .npy to skip OLS, or None to run OLS
save_xb_ols    = None   # path to save new xb_ols; None to skip
CUT_NUM_TEST   = 97     # head samples trimmed from the test stream by diff_and_filt_data
POST_CUT_TEST  = 80     # tail samples trimmed from the test stream by diff_and_filt_data
# ============================================================


# ----------- ROS1 bag reading --------------------------------

_typestore = get_typestore(Stores.ROS1_NOETIC)

def _read_topic(bag_path, topic):
    timestamps, msgs = [], []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            return timestamps, msgs
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            timestamps.append(timestamp * 1e-9)
            msgs.append(_typestore.deserialize_ros1(rawdata, connection.msgtype))
    return timestamps, msgs

def read_single_bag(bag_path, robot):
    joint_timestamps, joint_position, joint_velocity, joint_effort = [], [], [], []
    jaw_timestamps, jaw = [], []

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

    ts_list, msgs = _read_topic(bag_path, f'/{robot}/jaw/measured_js')
    for ts, msg in zip(ts_list, msgs):
        if len(msg.position) == 0:
            continue
        jaw_timestamps.append(ts)
        jaw.append([msg.position[0], msg.velocity[0], msg.effort[0]])

    print(f"  measured_js:     {len(joint_timestamps)} samples")
    print(f"  jaw measured_js: {len(jaw_timestamps)} samples")

    start_time = joint_timestamps[0]
    joint_timestamps = np.array(joint_timestamps) - start_time
    joints = np.column_stack((joint_timestamps, joint_position, joint_velocity, joint_effort))

    if len(jaw) > 0:
        jaw_timestamps = np.array(jaw_timestamps) - start_time
        jaw_mat = np.column_stack((jaw_timestamps, jaw))
        n = min(len(joints), len(jaw_mat))
        joints  = joints[:n]
        jaw_mat = jaw_mat[:n]
        t       = joints[:, 0]
        q_raw   = np.column_stack([joints[:, 1:1+expected_dof],                jaw_mat[:, 1:2]])
        dq_raw  = np.column_stack([joints[:, 1+expected_dof:1+2*expected_dof], jaw_mat[:, 2:3]])
        tau_raw = np.column_stack([joints[:, 1+2*expected_dof:],               jaw_mat[:, 3:4]])
    else:
        t       = joints[:, 0]
        q_raw   = joints[:, 1:1+expected_dof]
        dq_raw  = joints[:, 1+expected_dof:1+2*expected_dof]
        tau_raw = joints[:, 1+2*expected_dof:]

    return t, q_raw, dq_raw, tau_raw

def read_jacobian(bag_path, robot):
    topic = f'/{robot}/spatial/jacobian'
    ts_list, msgs = _read_topic(bag_path, topic)
    if not ts_list:
        raise RuntimeError(f"No messages on {topic}")
    N = len(msgs)
    J = np.zeros((6, 6, N))
    for i, msg in enumerate(msgs):
        J[:, :, i] = np.array(msg.data, dtype=float).reshape(6, 6)
    ts = np.array(ts_list) - ts_list[0]
    print(f"  spatial/jacobian: {N} samples")
    return ts, J

def read_wrench(bag_path, topic='/measured_cf'):
    ts_list, msgs = _read_topic(bag_path, topic)
    if not ts_list:
        raise RuntimeError(f"No messages on {topic}")
    w = np.zeros((len(msgs), 6))
    for i, msg in enumerate(msgs):
        w[i, 0] = msg.wrench.force.x
        w[i, 1] = msg.wrench.force.y
        w[i, 2] = msg.wrench.force.z
        w[i, 3] = msg.wrench.torque.x
        w[i, 4] = msg.wrench.torque.y
        w[i, 5] = msg.wrench.torque.z
    ts = np.array(ts_list) - ts_list[0]
    print(f"  {topic}: {len(msgs)} samples")
    return ts, w

# ----------- Read bags ---------------------------------------

print("\n=== Reading test bag ===")
_, q_raw_test, dq_raw_test, tau_raw_test = read_single_bag(test_bag, robot)

print("\n=== Reading test bag jacobian + GT force ===")
_, J_test_raw   = read_jacobian(test_bag, robot)
_, gt_force_raw = read_wrench(test_bag, '/measured_cf')

# ----------- Load model and trajectory params ----------------

robot_model = load_data(model_folder, model_name)
dof, fourier_order, base_freq, traj_optimizer_result, reg_norm_mat = load_data(traj_folder, traj_name)
print(f"\ndof={dof}  fourier_order={fourier_order}  base_freq={base_freq}")

h  = 1.0 / hz
fc = np.array([5.0]) * base_freq * fourier_order
print(f"fc = {fc}")

base_param_num = robot_model.base_num
H_b_func       = robot_model.H_b_func

# ----------- OLS (or load pre-computed) ---------------------

if xb_ols_path is not None:
    print(f"\n=== Loading pre-computed OLS result from {xb_ols_path} ===")
    xb_ols = np.load(xb_ols_path)
else:
    print("\n=== Reading train bag ===")
    _, q_raw_train, dq_raw_train, tau_raw_train = read_single_bag(train_bag, robot)
    t_train = np.arange(len(q_raw_train), dtype=float) / hz
    t_cut_train, q_f_train, dq_f_train, ddq_f_train, tau_f_train, _, _ = \
        diff_and_filt_data(dof, h, t_train, q_raw_train, dq_raw_train, tau_raw_train,
                           fc, fc, fc, fc, cut_num=98, post_cut=10)
    print(f'train: raw={len(tau_raw_train)}  after cut={len(tau_f_train)}')

    W_b_train, tau_s_train = gen_regressor(base_param_num, H_b_func,
                                           q_f_train, dq_f_train, ddq_f_train, tau_f_train)
    print('\n=== Running OLS ===')
    t0 = time.time()
    xb_ols = np.linalg.lstsq(W_b_train, tau_s_train, rcond=None)[0]
    print(f'OLS done in {time.time()-t0:.1f}s')

    if save_xb_ols is not None:
        np.save(save_xb_ols, xb_ols)
        print(f'xb_ols saved to {save_xb_ols}')

# ----------- Filter test data --------------------------------

t_test = np.arange(len(q_raw_test), dtype=float) / hz
t_cut_test, q_f_test, dq_f_test, ddq_f_test, tau_f_test, _, _ = \
    diff_and_filt_data(dof, h, t_test, q_raw_test, dq_raw_test, tau_raw_test,
                       fc, fc, fc, fc, cut_num=CUT_NUM_TEST, post_cut=POST_CUT_TEST)
print(f'test:  raw={len(tau_raw_test)}  after cut={len(tau_f_test)}')

# ----------- Force estimation --------------------------------

W_b_test, tau_s_test = gen_regressor(base_param_num, H_b_func,
                                     q_f_test, dq_f_test, ddq_f_test, tau_f_test)
tau_ps_ols_test = W_b_test.dot(xb_ols)
tau_p_ols_test  = np.zeros(tau_f_test.shape)
for i in range(dof):
    tau_p_ols_test[:, i] = tau_ps_ols_test[i::dof]

estimated_external_tau = tau_f_test - tau_p_ols_test












print(f'\nestimated_external_tau shape: {estimated_external_tau.shape}')

# ----------- Map joint torque -> Cartesian wrench ------------

N_test_raw = J_test_raw.shape[2]
J_test = J_test_raw[:, :, CUT_NUM_TEST : N_test_raw - POST_CUT_TEST]

N = min(J_test.shape[2], estimated_external_tau.shape[0])
J_test                 = J_test[:, :, :N]
estimated_external_tau = estimated_external_tau[:N]
t_cut_test_aligned     = t_cut_test[:N]

fs_diff  = estimated_external_tau[:, :6].T   # (6, N)
fs_force = np.zeros((6, N))

# transpose matrix

# angle_x = np.deg2rad(45)
# Rx_45 = np.array([[1, 0,               0              ],
#                   [0, np.cos(angle_x), -np.sin(angle_x)],
#                   [0, np.sin(angle_x),  np.cos(angle_x)]])

# angle_z = np.deg2rad(-30)
# Rz_x = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
#                  [np.sin(angle_z),  np.cos(angle_z), 0],
#                  [0,                0,               1]])

# flip_y_axis   = np.diag([-1.0, -1.0, 1.0])
# R_robot_to_gt = flip_y_axis @ Rx_45.T @ Rz_x.T

flip_y_axis = np.diag([-1, -1, 1])

# New basis:
# new x = old z
# new y = old x
# new z = old y
basis = np.array([[ 0, 0, -1],
               [ 0, 1,  0],
               [ 1, 0,  0]])

x_deg = -30
angle_z = np.deg2rad(x_deg)

Rz_x = np.array([
    [np.cos(angle_z), -np.sin(angle_z), 0],
    [np.sin(angle_z),  np.cos(angle_z), 0],
    [0,                0,               1]
])

Rx_minus_45 = np.array([
    [1, 0,       0      ],
    [0, 0.7071,  0.7071 ],
    [0, -0.7071, 0.7071 ]
])

rotated = Rz_x @ Rx_minus_45 @ flip_y_axis

T_transpose = np.linalg.inv(rotated) @ basis

for i in range(N):
    Ji = J_test[:, :, i]
    fs_force[:, i] = np.linalg.pinv(Ji.T) @ fs_diff[:, i]
    fs_force[:3, i] = fs_force[:3, i] @ T_transpose

# after the loop
axis_correction = np.array([[ 0, 1,  0],
                              [-1,  0,  0],
                              [ 0,  0, 1]], dtype=float)
fs_force[:3, :] = axis_correction @ fs_force[:3, :]   # (3,3)@(3,N)

fs_force = fs_force.T   # (N, 6)

windowSize = 50
b = (1.0 / windowSize) * np.ones(windowSize)
fs_force = lfilter(b, [1.0], fs_force, axis=0)

N_gt_raw = gt_force_raw.shape[0]
gt_force = gt_force_raw[CUT_NUM_TEST : N_gt_raw - POST_CUT_TEST]
gt_force = gt_force[:N]
print(f'fs_force shape: {fs_force.shape}, gt_force shape: {gt_force.shape}')

# ----------- Plot --------------------------------------------

plot_meas_pred_tau(t_cut_test, tau_f_test, tau_p_ols_test,
                   robot_model.coordinates_joint_type, robot_model.coordinates)

fig, axes = plt.subplots(dof, 1, figsize=(12, 2 * dof), sharex=True)
if dof == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    ax.plot(t_cut_test, estimated_external_tau[:, i])
    ax.set_ylabel(f'Joint {i+1}')
    ax.grid(True)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Estimated External Torque (OLS)')
plt.tight_layout()

rmse   = np.sqrt(np.mean((fs_force[:, :3] - gt_force[:, :3]) ** 2, axis=0))
labels = ['Fx', 'Fy', 'Fz']

fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
fig.suptitle('Force Estimation (OLS), dVRK PSM1')
for k, ax in enumerate(axes):
    ax.plot(t_cut_test_aligned, gt_force[:, k], 'b', label='measured')
    ax.plot(t_cut_test_aligned, fs_force[:, k], 'r', label='predicted')
    ax.set_title(f'{labels[k]},  RMSE = {rmse[k]:.4f}')
    ax.set_ylabel('Force / N')
    ax.grid(True)
axes[-1].set_xlabel('Time (s)')
axes[0].legend(loc='upper right')

plt.show()
