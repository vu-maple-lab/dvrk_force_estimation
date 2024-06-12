#!/usr/bin/env python3
import json

# derived from dvrk_bag_replay

import crtk
import os
import sys
import time
import signal

import numpy as np
import rospy
import rosbag
import numpy
import PyKDL
import argparse
import subprocess


# simplified arm class to replay motion, better performance than
# dvrk.psm since we're only subscribing to topics we need
class arm_custom:

    # simplified jaw class to close gripper
    class __Jaw:
        def __init__(self, ros_namespace, expected_interval, operating_state_instance):
            self.__crtk_utils = crtk.utils(self, ros_namespace, expected_interval, operating_state_instance)
            self.__crtk_utils.add_move_jp()
            self.__crtk_utils.add_servo_jp()

    def __init__(self, device_namespace, expected_interval):
        # ROS initialization
        if not rospy.get_node_uri():
            rospy.init_node('simplified_arm_class', anonymous = True, log_level = rospy.WARN)
        # populate this class with all the ROS topics we need
        self.crtk_utils = crtk.utils(self, device_namespace, expected_interval)
        self.crtk_utils.add_operating_state()
        self.crtk_utils.add_servo_jp()
        self.crtk_utils.add_move_jp()
        self.jaw = self.__Jaw(device_namespace + '/jaw', expected_interval, operating_state_instance = self)



# helper function to kill the rosbag process
def terminate_process_and_children(p):
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % p.pid, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    assert retcode == 0, "ps command returned %d" % retcode
    print(ps_output)
    ps_output = ps_output.decode('utf-8')
    print(ps_output.split("\n")[:-1])
    for pid_str in ps_output.split("\n")[:-1]:
        os.kill(int(pid_str), signal.SIGINT)
    p.terminate()

# if sys.version_info.major < 3:
#     input = raw_input

# ---------------------------------------------
# ros setup
# ---------------------------------------------
# ros init node
rospy.init_node('dvrk_data_collection', anonymous=True)
# strip ros arguments
argv = rospy.myargv(argv=sys.argv)

# ---------------------------------------------
# parse arguments
# ---------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('-a', '--arm', type=str, required=True,
#                     choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
#                     help='arm name corresponding to ROS topics without namespace.')
parser.add_argument('-a', '--arm', type=str, default='PSM1',
                    choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                    help='arm name corresponding to ROS topics without namespace.')
# parser.add_argument('-f', '--csv', type=argparse.FileType('r'), required=True,
#                     help='csv file containing the trajectory to replay.')
parser.add_argument('-f', '--file_json', type=str, default='three_wpi_traj.json',
                    help='json file containing the trajectory to replay.')
# parser.add_argument('-s', '--setup', type=str, required=True,
#                     choices=['PSM', 'PSM-SUJ'],
#                     help='description of your dvrk setup, necessary for collect shell script selection.')
parser.add_argument('-s', '--setup', type=str, default='PSM',
                    choices=['PSM', 'PSM-SUJ'],
                    help='description of your dvrk setup, necessary for collect shell script selection.')
parser.add_argument('-c', '--config', type=str, required=False, default='1',
                    help='index of your current configuration, if you changed SUJ position, please use another index. '
                         'e.g., \"-c 1\" for configuration 1 and \"-c 2\" for for configuration 1')


args = parser.parse_args(argv[1:])  # skip argv[0], script name

# ---------------------------------------------
# prepare rosbag script
# ---------------------------------------------
# create data folder
if not os.path.exists(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

# create configuration sub_folder
folder_name = os.path.join(os.getcwd(), 'data')
sub_folder_name = os.path.join(os.getcwd(), os.path.join('data', (args.config)))
if not os.path.exists(sub_folder_name):
    os.mkdir(sub_folder_name)

# determine script based on setup
PSM_id = args.arm.strip('PSM')
if args.setup == 'PSM':
#   command = "rosbag record -O {0}_{1}.bag /{1}/tool_type /{1}/measured_cp /{1}/measured_js /{1}/jaw/measured_js /{1}/spatial/jacobian".format(os.path.join(sub_folder_name,
#                                                                                                                                                            os.path.basename(args.csv.name).split('.csv')[0]),
#   
    command = "rosbag record -O {0}_{1}.bag /{1}/measured_js /{1}/jaw/measured_js ".format(os.path.join(sub_folder_name, args.file_json.split('.')[0]),'PSM'+str(PSM_id))
    
    # sanity check command
    sanity_cmd = "rosbag info {0}_{1}.bag".format(os.path.join(sub_folder_name, args.file_json.split('.')[0]), 'PSM'+str(PSM_id))
else:
    sys.exit('-- Unsupported setup')

# ---------------------------------------------
# read commanded joint position
# ---------------------------------------------
poses = []
# with open(args.csv.name) as csvfile:
#     for pose in csv.reader(csvfile, delimiter=','):
#         poses.append(np.array(pose).astype(float))
#     csvfile.close()


f = open(args.file_json)

data = json.load(f)

poses_origin = data["js"]

poses = []
# jaw_poses= []

for pose_old in poses_origin:
    poses.append(np.array(pose_old[:-1]))
    # jaw_poses.append(np.array(pose_old[-1]))

f.close()


# ---------------------------------------------
# prepare psm
# ---------------------------------------------
print('-- This script will replay a trajectory defined in %s on arm %s' % (args.file_json, args.arm))

# create arm
arm = arm_custom(device_namespace=args.arm, expected_interval=0.001)

# make sure the arm is powered
print('-- Enabling arm')
if not arm.enable(10):
    sys.exit('-- Failed to enable within 10 seconds')

print('-- Homing arm')
if not arm.home(10):
    sys.exit('-- Failed to home within 10 seconds')

input('---> Make sure the arm is ready to move using joint positions\n     You need to have a tool/instrument in place and properly engaged\n     Press "Enter" when the arm is ready')


# close gripper
input('---> Press \"Enter\" to close the instrument\'s jaws')

##################################################
jaw_jp = np.array([-20.0 * np.pi / 180.0])
arm.jaw.move_jp(jaw_jp).wait()

# go to initial position and wait
input('---> Press \"Enter\" to move to start position')
jp = poses[0]
# jaw_jp = jaw_poses[0]
jaw_jp = np.array([0])
arm.move_jp(jp).wait()
arm.jaw.move_jp(jaw_jp).wait()

# ---------------------------------------------
# start playing trajectory and data collection
# ---------------------------------------------
# play trajectory
input('---> Press \"Enter\" to replay the recorded trajectory and collect data')

# run shell script
# rosbag_process = subprocess.Popen(command.split(' '))
time.sleep(2)

# main play process
counter = 0
total = len(poses)
start_time = time.time()

# for pose in poses:
for i in range(len(poses)):
    start_t = time.time()
    # arm.jaw.servo_jp(jaw_poses[i]*0.1)
    jaw_jp = np.array([-0.01])
    # jaw_jp = np.array([0.03])
    # arm.jaw.move_jp(jaw_jp).wait()
    arm.jaw.servo_jp(jaw_jp)
    time.sleep(0.5)
    arm.servo_jp(poses[i])
    counter = counter + 1
    # report progress
    sys.stdout.write('\r-- Progress %02.1f%%' % (float(counter) / float(total) * 100.0))
    sys.stdout.flush()
    end_t = time.time()
    delta_t = 0.005 - (end_t - start_t)        ###############################check console rate
    # if process takes time larger than console rate, don't sleep
    if delta_t > 0:
        time.sleep(delta_t)

# stop bagging
# terminate_process_and_children(rosbag_process)

print('\n--> Time to replay trajectory: %f seconds' % (time.time() - start_time))
print('--> Done!')

# check if required topics are collected correctly
os.system(sanity_cmd)