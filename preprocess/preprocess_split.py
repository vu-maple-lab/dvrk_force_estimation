import os
import sys
from os.path import join, exists
import numpy as np
from scipy import interpolate
from pathlib import Path

# ── Usage ─────────────────────────────────────────────────────────────────────
# python preprocess_split.py <input_folder> <output_root> [train_ratio] [val_ratio] [test_ratio]
# Example:
#   python preprocess_split.py ../Data/All_trajectory_Collection/ ../Data/All_trajectory_Collection/data_7_2_1/
#   python preprocess_split.py ../Data/All_trajectory_Collection/ ../Data/All_trajectory_Collection/data_6_3_1/ 0.6 0.3 0.1
#
# Input folder must contain: joints/, jacobian/, jaw/
# Files with sensor data must also exist in: sensor/
# Output: <output_root>/train/, val/, test/
# ─────────────────────────────────────────────────────────────────────────────

#python preprocess_split.py ../Data/All_trajectory_Collection/ ../Data/All_trajectory_Collection/data_7_2_1/


input_folder  = sys.argv[1]
output_root   = sys.argv[2]
cut_off       = 100

TRAIN_RATIO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
VAL_RATIO   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
TEST_RATIO  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_concat(file_list, data_path):
    """Load and concatenate CSVs, making timestamps continuous."""
    all_data = np.array([])
    for fname in file_list:
        data = np.loadtxt(join(data_path, fname), delimiter=',')
        end_idx = int(data.shape[0] / cut_off)
        data = data[:end_idx * cut_off, :]
        if data[0, 0] > 1000:
            data[:, 0] -= data[0, 0]
        if all_data.size:
            data[:, 0] = all_data[-1, 0] + data[:, 0] + 0.001
        all_data = np.vstack((all_data, data)) if all_data.size else data
    return all_data


def interp_cols(src_time, src_data, target_time, n_cols):
    out = np.zeros((len(target_time), n_cols + 1))
    out[:, 0] = target_time
    for i in range(n_cols):
        f = interpolate.interp1d(src_time, src_data[:, i + 1])
        out[:, i + 1] = f(target_time)
    return out


def process_split(file_list, sensor_files_set, split_name):
    out_path = join(output_root, split_name)
    for sub in ['joints', 'jacobian', 'jaw', 'sensor']:
        Path(join(out_path, sub)).mkdir(parents=True, exist_ok=True)

    joint_path    = join(input_folder, 'joints')
    jacobian_path = join(input_folder, 'jacobian')
    jaw_path      = join(input_folder, 'jaw')
    sensor_path   = join(input_folder, 'sensor')

    all_joints   = load_and_concat(file_list, joint_path)
    all_jacobian = load_and_concat(file_list, jacobian_path)
    all_jaw      = load_and_concat(file_list, jaw_path)

    sensor_in_split = [f for f in file_list if f in sensor_files_set]
    all_sensor = load_and_concat(sensor_in_split, sensor_path) if sensor_in_split else None

    # Common time range
    joint_time    = all_joints[:, 0]   - all_joints[0, 0]
    jacobian_time = all_jacobian[:, 0] - all_joints[0, 0]
    jaw_time      = all_jaw[:, 0]      - all_joints[0, 0]

    t_start = max(joint_time[0], jacobian_time[0], jaw_time[0])
    t_end   = min(joint_time[-1], jacobian_time[-1], jaw_time[-1])

    if all_sensor is not None:
        sensor_time = all_sensor[:, 0] - all_joints[0, 0]
        t_start = max(t_start, sensor_time[0])
        t_end   = min(t_end,   sensor_time[-1])

    t = np.arange(t_start, t_end, 0.05)

    # Interpolate and save
    np.savetxt(
        join(out_path, 'joints', 'interpolated_all_joints.csv'),
        interp_cols(joint_time, all_joints, t, 18), delimiter=','
    )
    np.savetxt(
        join(out_path, 'jacobian', 'interpolated_all_jacobian.csv'),
        interp_cols(jacobian_time, all_jacobian, t, 36), delimiter=','
    )
    np.savetxt(
        join(out_path, 'jaw', 'interpolated_all_jaw.csv'),
        interp_cols(jaw_time, all_jaw, t, 3), delimiter=','
    )
    if all_sensor is not None:
        np.savetxt(
            join(out_path, 'sensor', 'interpolated_all_sensor.csv'),
            interp_cols(sensor_time, all_sensor, t, 6), delimiter=','
        )
        print(f'  [{split_name}] sensor saved ({len(sensor_in_split)} files with sensor)')
    else:
        print(f'  [{split_name}] no sensor data in this split')

    print(f'  [{split_name}] {len(file_list)} files -> {len(t)} timesteps -> {out_path}')


# ── Discover files ────────────────────────────────────────────────────────────

joint_path  = join(input_folder, 'joints')
sensor_path = join(input_folder, 'sensor')

all_files    = sorted([f for f in os.listdir(joint_path) if f.endswith('.csv')])
sensor_files = set(os.listdir(sensor_path)) if exists(sensor_path) else set()

files_with_sensor    = [f for f in all_files if f in sensor_files]
files_without_sensor = [f for f in all_files if f not in sensor_files]

N      = len(all_files)
n_test = max(1, round(N * TEST_RATIO))
n_val  = max(1, round(N * VAL_RATIO))

print(f'Total files: {N}')
print(f'Files with sensor:    {files_with_sensor}')
print(f'Files without sensor: {files_without_sensor}')
print(f'Planned split -> train: {N - n_val - n_test}, val: {n_val}, test: {n_test}')

if len(files_with_sensor) < n_test:
    raise ValueError(
        f'Need {n_test} file(s) with sensor for test set, '
        f'but only {len(files_with_sensor)} available: {files_with_sensor}'
    )

# Test: first n_test files that have sensor
test_files = files_with_sensor[:n_test]

# Remaining: everything else (preserve sorted order)
remaining = [f for f in all_files if f not in set(test_files)]
val_files   = remaining[:n_val]
train_files = remaining[n_val:]

print(f'\nAssigned:')
print(f'  train : {train_files}')
print(f'  val   : {val_files}')
print(f'  test  : {test_files}')
print()

# ── Process each split ────────────────────────────────────────────────────────

process_split(train_files, sensor_files, 'train')
process_split(val_files,   sensor_files, 'val')
process_split(test_files,  sensor_files, 'test')

print('\nDone.')
