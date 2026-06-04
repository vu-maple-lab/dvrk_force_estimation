import sys
import torch
from network import *
import utils
from pathlib import Path
import numpy as np
from scipy import interpolate
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import indirectDataset

#python test_force.py my_exp lstm seal PSM1 ../Data/All_trajectory_Collection/data_7_2_1/test/


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
data = 'free_space'
epoch_to_use = 0
exp  = sys.argv[1]
net  = sys.argv[2]
seal = sys.argv[3]
arm  = sys.argv[4]
test_data_folder = sys.argv[5] if len(sys.argv) > 5 else '../Data/test_csv/'

JOINTS = utils.JOINTS
window = 1000
root = Path('..')
in_joints = [0, 1, 2, 3, 4, 5]
preprocess = 'filtered_torque_colon_9_26'

if seal == 'seal':
    fs = 'free_space'
else:
    fs = 'no_cannula'

max_torque = torch.tensor(utils.max_torque).to(device)
ATTN_nhead = 1
print('device is:', device)

# Coordinate transform
flip_y_axis  = np.diag([-1.0, -1.0, 1.0])
basis        = np.array([[ 0,  0, -1],
                          [ 0,  1,  0],
                          [ 1,  0,  0]], dtype=float)
angle_z      = np.deg2rad(-22)
Rz_x         = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                          [np.sin(angle_z),  np.cos(angle_z), 0],
                          [0,                0,               1]], dtype=float)
Rx_minus_45  = np.array([[1,  0,      0     ],
                          [0,  0.7071, 0.7071],
                          [0, -0.7071, 0.7071]], dtype=float)
rotated      = Rz_x @ Rx_minus_45 @ flip_y_axis
T_transpose  = np.linalg.inv(rotated) @ basis

axis_correction = np.array([[ 0,  1,  0],
                              [-1,  0,  0],
                              [ 0,  0,  1]], dtype=float)


def apply_coord_transform(force_np):
    """Apply robot->sensor coordinate transform to force columns (N,3)."""
    f = force_np @ T_transpose
    f = (axis_correction @ f.T).T
    return f


def main():
    path = test_data_folder

    dataset = indirectDataset(path, window, utils.SKIP, in_joints, is_rnn=True)
    loader  = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model_root = []
    for j in range(JOINTS):
        model_root.append(root / preprocess / net / arm / (fs + str(j)))
        print('Loading model from:', model_root[j])

    networks = []
    for j in range(JOINTS):
        if net == 'lstm':
            networks.append(torqueLstmNetwork(batch_size, device).to(device))
        elif net == 'attn':
            networks.append(torqueTransNetwork(batch_size, device, attn_nhead=ATTN_nhead).to(device))
        else:
            networks.append(fsNetwork(window).to(device))
        utils.load_prev(networks[j], model_root[j], epoch_to_use)
        print('Loaded model', j)

    all_diff     = torch.tensor([])
    all_jacobian = torch.tensor([])
    all_time     = torch.tensor([])

    for position, velocity, torque, jacobian, time in loader:
        position = position.to(device)
        velocity = velocity.to(device)
        posvel   = torch.cat((position, velocity), axis=2).contiguous()

        time     = time.permute((1, 0)).squeeze(-1)   # (window,)
        torque   = torque.squeeze(0)                   # (window, 6)
        jacobian = jacobian.squeeze(0)                 # (window, 36)

        step_pred = torch.zeros(torque.size())
        for j in range(JOINTS):
            pred = networks[j](posvel).squeeze().detach()  # (window,)
            pred = pred * max_torque[j]
            step_pred[:, j] = pred.cpu()

        diff = torque - step_pred  # tau_measured - tau_predicted

        all_diff     = torch.cat((all_diff,     diff),     axis=0) if all_diff.size()     else diff
        all_jacobian = torch.cat((all_jacobian, jacobian), axis=0) if all_jacobian.size() else jacobian
        all_time     = torch.cat((all_time,     time.cpu()), axis=0) if all_time.size()   else time.cpu()

    all_force_t  = utils.calculate_force(all_jacobian, all_diff)   # (N, 6)
    all_force_np = all_force_t.numpy().copy()

    # Coordinate transform
    all_force_np[:, :3] = apply_coord_transform(all_force_np[:, :3])
    
    b = (1.0 / 50) * np.ones(50)
    all_force_np = lfilter(b, [1.0], all_force_np, axis=0)

    pred_time = all_time.numpy()

    # save CSV
    result    = np.hstack([pred_time[:, None], all_force_np])
    out_path  = Path(test_data_folder) / (net + '_' + seal + '_force_' + preprocess + '.csv')
    np.savetxt(str(out_path), result)
    print('Predicted force saved to:', out_path)

    # ground truth
    sensor_csv   = Path(test_data_folder) / 'sensor' / 'interpolated_all_sensor.csv'
    sensor_data  = np.loadtxt(str(sensor_csv), delimiter=',')
    sensor_time  = sensor_data[:, 0]
    sensor_force = sensor_data[:, 1:4]   # fx, fy, fz

    # Align to common time range
    t_start = max(pred_time[0],  sensor_time[0])
    t_end   = min(pred_time[-1], sensor_time[-1])
    mask    = (pred_time >= t_start) & (pred_time <= t_end)

    pred_time_aligned  = pred_time[mask]
    pred_force_aligned = all_force_np[mask, :3]   # fx, fy, fz only

    sensor_interp = np.zeros((mask.sum(), 3))
    for i in range(3):
        f = interpolate.interp1d(sensor_time, sensor_force[:, i])
        sensor_interp[:, i] = f(pred_time_aligned)

    # RMSE
    rmse   = np.sqrt(np.mean((pred_force_aligned - sensor_interp) ** 2, axis=0))
    labels = ['Fx', 'Fy', 'Fz']
    print('\nForce RMSE (N):')
    for k, lb in enumerate(labels):
        print(f'  {lb}: {rmse[k]:.4f}')
    print(f'  mean: {np.mean(rmse):.4f}')

    # Save comparison CSV
    comparison = np.hstack([pred_time_aligned[:, None], pred_force_aligned, sensor_interp])
    cmp_path   = Path(test_data_folder) / (net + '_' + seal + '_force_comparison_' + preprocess + '.csv')
    np.savetxt(str(cmp_path), comparison)
    print('Comparison saved to:', cmp_path)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f'Force Estimation ({net.upper()}), dVRK {arm}')
    for k, ax in enumerate(axes):
        ax.plot(pred_time_aligned, sensor_interp[:, k],    'b', label='measured')
        ax.plot(pred_time_aligned, pred_force_aligned[:, k], 'r', label='predicted')
        ax.set_title(f'{labels[k]},  RMSE = {rmse[k]:.4f}')
        ax.set_ylabel('Force / N')
        ax.grid(True)
    axes[-1].set_xlabel('Time (s)')
    axes[0].legend(loc='upper right')

    fig_path = Path(test_data_folder) / (net + '_' + seal + '_force_comparison.png')
    plt.savefig(str(fig_path), dpi=150)
    print('Plot saved to:', fig_path)
    plt.show()


if __name__ == "__main__":
    main()
