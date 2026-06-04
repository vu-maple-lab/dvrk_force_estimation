import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# python plot_torque.py <net> <seal> [data_folder]
# Default:  python plot_torque.py lstm seal
# Custom:   python plot_torque.py lstm seal ../Data/train_csv/

net  = sys.argv[1]
seal = sys.argv[2]
data_folder = sys.argv[3] if len(sys.argv) > 3 else '../Data/train_csv/'

preprocess = 'filtered_torque_colon_9_26'

# ── Load data ─────────────────────────────────────────────────────────────────
joint_csv = Path(data_folder) / 'joints' / 'interpolated_all_joints.csv'
pred_csv  = Path(data_folder) / f'{net}_{seal}_pred_{preprocess}.csv'

joint_data = np.loadtxt(str(joint_csv), delimiter=',')
pred_data  = np.loadtxt(str(pred_csv))

# Time (relative, starting from 0)
time_joint = joint_data[:, 0] - joint_data[0, 0]
time_pred  = pred_data[:, 0]

# Measured torque: columns 13-18 (0-indexed) = joints 1-6
measured = joint_data[:, 13:19]

# Predicted torque: columns 1-6 (0-indexed)
predicted = pred_data[:, 1:7]

# Align lengths (pred may be shorter due to windowing)
N = min(len(time_pred), len(time_joint))
time_pred  = time_pred[:N]
measured   = measured[:N]
predicted  = predicted[:N]

# ── RMSE ──────────────────────────────────────────────────────────────────────
rmse_per_joint = np.sqrt(np.mean((measured - predicted) ** 2, axis=0))  # (6,)
rmse_overall   = np.mean(rmse_per_joint)

print(f'Overall RMSE: {rmse_overall:.4f}')
for j in range(6):
    print(f'  Joint {j+1}: {rmse_per_joint[j]:.4f}')

# ── Plot (2×3 grid, same style as MATLAB free.mlx / force_est_OLS.py) ────────
fig, axes = plt.subplots(2, 3, figsize=(14, 6), constrained_layout=True)
fig.suptitle(f'Torque Prediction ({net.upper()}), RMSE = {rmse_overall:.4f}')

for j, ax in enumerate(axes.flat):
    ax.plot(time_pred, measured[:, j],   'b', label='measured',  linewidth=0.8)
    ax.plot(time_pred, predicted[:, j],  'r', label='predicted', linewidth=0.8)
    ax.set_title(f'Joint {j+1},  RMSE = {rmse_per_joint[j]:.4f}')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Torque / Nm')
    ax.grid(True)

axes[0, 0].legend(loc='upper right')

fig_path = Path(data_folder) / f'{net}_{seal}_torque_comparison.png'
plt.savefig(str(fig_path), dpi=150)
print(f'Plot saved to: {fig_path}')
plt.show()
