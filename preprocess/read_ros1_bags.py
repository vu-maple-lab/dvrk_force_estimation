import os
from pathlib import Path
import numpy as np
import time
import argparse
from scipy import interpolate

# rosbags: pure Python, no ROS installation needed (pip install rosbags)
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

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


class RosbagParser():

    def __init__(self, args):
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

    def interp(self, time, mat):
        new_mat = np.zeros((len(time), mat.shape[1]))
        new_mat[:, 0] = time
        for i in range(mat.shape[1]):
            f = interpolate.interp1d(mat[:, 0], mat[:, i])
            new_mat[:, i] = f(time)
        return new_mat

    def single_datapoint_processing(self, file_name):
        bag_path = self.folder + file_name

        force_sensor = []
        force_sensor_timestamps = []
        joint_position = []
        joint_velocity = []
        joint_effort = []
        joint_timestamps = []
        jacobian = []
        jacobian_timestamps = []
        jaw = []
        jaw_timestamps = []

        print("Processing " + file_name)

        # --- measured_js ---
        ts_list, msgs = _read_topic(bag_path, '/PSM1/measured_js')
        for ts, msg in zip(ts_list, msgs):
            if len(msg.position) == 0:
                continue
            joint_timestamps.append(ts)
            joint_velocity.append(list(msg.velocity))
            joint_position.append(list(msg.position))
            joint_effort.append(list(msg.effort))

        # --- spatial jacobian ---
        ts_list, msgs = _read_topic(bag_path, '/PSM1/spatial/jacobian')
        for ts, msg in zip(ts_list, msgs):
            jacobian_timestamps.append(ts)
            jacobian.append(list(msg.data))

        # --- jaw measured_js ---
        ts_list, msgs = _read_topic(bag_path, '/PSM1/jaw/measured_js')
        for ts, msg in zip(ts_list, msgs):
            if len(msg.position) == 0:
                continue
            jaw_timestamps.append(ts)
            jaw.append([msg.position[0], msg.velocity[0], msg.effort[0]])

        # --- force sensor ---
        ts_list, msgs = _read_topic(bag_path, '/measured_cf')
        for ts, msg in zip(ts_list, msgs):
            force_sensor_timestamps.append(ts)
            force_sensor.append([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ])

        print("Processed wrench: counts: {}".format(len(force_sensor_timestamps)))
        print("Processed state joint current: count: {}".format(len(joint_timestamps)))
        print("Processed state jaw current: count: {}".format(len(jaw_timestamps)))
        print("Processed Jacobian: count: {}".format(len(jacobian)))

        # --- create output directories ---
        try:
            (Path(self.output) / "joints").mkdir(mode=0o777, parents=False)
        except OSError:
            print("Joints path exists")

        try:
            (Path(self.output) / "jacobian").mkdir(mode=0o777, parents=False)
        except OSError:
            print("Jacobian path exists")

        if len(force_sensor) > 0:
            try:
                (Path(self.output) / "sensor").mkdir(mode=0o777, parents=False)
            except OSError:
                print("Sensor path exists")

        if len(jaw) > 0:
            try:
                (Path(self.output) / "jaw").mkdir(mode=0o777, parents=False)
            except OSError:
                print("Jaw path exists")

        # --- time normalization and array construction ---
        start_time = joint_timestamps[0]
        joint_timestamps = np.array(joint_timestamps) - start_time
        jacobian_timestamps = np.array(jacobian_timestamps) - start_time
        joints = np.column_stack((joint_timestamps, joint_position, joint_velocity, joint_effort))
        jacobian = np.column_stack((jacobian_timestamps, jacobian))

        if len(force_sensor) > 0:
            force_sensor_timestamps = np.array(force_sensor_timestamps) - start_time
            force_sensor = np.column_stack((force_sensor_timestamps, force_sensor))
        else:
            force_sensor = None

        if len(jaw) > 0:
            jaw_timestamps = np.array(jaw_timestamps) - start_time
            jaw = np.squeeze(np.array(jaw))
            jaw = np.column_stack((jaw_timestamps, jaw))
            if len(joints) < len(jaw):
                jaw = jaw[0:len(joints), :]
            else:
                joints = joints[0:len(jaw), :]

        # --- save CSVs ---
        out_name = self.prefix + str(self.index)
        np.savetxt(self.output + "joints/" + out_name + ".csv", joints, delimiter=',')
        np.savetxt(self.output + "jacobian/" + out_name + ".csv", jacobian, delimiter=',')
        if force_sensor is not None:
            np.savetxt(self.output + "sensor/" + out_name + ".csv", force_sensor, delimiter=',')
        if len(jaw) > 0:
            np.savetxt(self.output + "jaw/" + out_name + ".csv", jaw, delimiter=',')
        print("Wrote out " + out_name)
        print("")

    def parse_bags(self):
        print("\nParsing\n")
        files = sorted(os.listdir(self.folder))
        for file_name in files:
            if file_name.endswith('.bag'):
                self.single_datapoint_processing(file_name)
                self.index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='../data/', type=str, help='Path to Rosbag folder')
    parser.add_argument('-o', '--output', default='./parsed_data/', type=str, help='Path to write out parsed csv')
    parser.add_argument('--prefix', default='free_train_', type=str, help='Prefix for output csv names')
    parser.add_argument('--index', default=0, type=int, help='Starting index for output csv names')
    args = parser.parse_args()
    start = time.time()
    rosbag_parser = RosbagParser(args)
    rosbag_parser.parse_bags()
    print("Parsing complete")
    print("The entire process takes {} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()

# example: python3 read_ros1_bags.py -f ~/data/bags/ -o ~/data/parsed/ --prefix trial_ --index 0
