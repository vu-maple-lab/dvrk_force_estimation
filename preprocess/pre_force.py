import os
import sys
from os.path import join
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

all_sensor = np.array([])


split = sys.argv[1]

path = join('..', 'simon_trocar_feb_27', split)

sensor_path = join(path, 'sensor')
cut_off = 100

for cur_file in os.listdir(sensor_path):
    joints = np.loadtxt(join(sensor_path, cur_file), delimiter=',')
    end_idx = int(joints.shape[0] / cut_off)
    joints = joints[0:end_idx * cut_off, :]
    if joints[0, 0] > 1000:
        joints[:, 0] = joints[:, 0] - joints[0, 0]
    if all_sensor.size:
        joints[:, 0] = all_sensor[-1, 0] + joints[:, 0] + 0.001
    all_sensor = np.vstack((all_sensor, joints)) if all_sensor.size else joints


print(all_sensor.shape)
sensor_time = all_sensor[:, 0] - all_sensor[0, 0]

start_time = np.max([sensor_time[0]])
end_time = np.min([sensor_time[-1]])
print(start_time, end_time)

interpolated_time = np.arange(start_time, end_time, 0.05)
interp_sensor = np.zeros((interpolated_time.shape[0], 7))
interp_sensor[:, 0] = interpolated_time

for i in range(6):
    f = interpolate.interp1d(sensor_time, all_sensor[:, i + 1])
    interp_sensor[:, i + 1] = f(interpolated_time)

np.savetxt(join(sensor_path, 'interpolated_all_sensor.csv'), interp_sensor, delimiter=',')
