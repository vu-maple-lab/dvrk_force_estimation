import numpy as np
import time
import sympy
from dvrk_dynamic_identification.robot_def import RobotDef
from dvrk_dynamic_identification.kinematics.geometry import Geometry
from dvrk_dynamic_identification.dynamics.dynamics import Dynamics
from dvrk_dynamic_identification.utils import new_sym
from dvrk_dynamic_identification.utils import save_data
from dvrk_dynamic_identification.robot_model_data import RobotModel
import os
import sys
dynamic_path = os.path.abspath(__file__+"/../../") # move to the parent folder [dvrk_force_estimation]
# print(dynamic_path)
sys.path.append(dynamic_path)
import matplotlib.pyplot as plt
from IPython.display import display
import argparse

def create_robot_model(model_name:str)->RobotDef:
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = new_sym('q:11')
    _pi = sympy.pi

    dh = []
    springs = []
    friction_type = []

    ### Set parameters, based on the model in the manuscript
    ### FYI, we are using the simplified model, please check the manuscript for more information
    qmd5 = 1.0186 * q5
    qmd6 = -0.8306 * q5 + 1.2178 * q6
    qmd7 = -0.8306 * q5 + 1.2178 * q7

    l_2L1 = 96 * 0.001
    l_2L2 = 516 * 0.001
    l_2L3 = 40.09 * 0.001

    l_2H1 = 144.54 * 0.001
    l_2H2 = 38.08 * 0.001

    l_3 = 40.09 * 0.001

    l_RCC = 431.8 * 0.001
    l_tool = 416.2 * 0.001
    l_p2y = 9.1 * 0.001

    L_b = 0
    L_1 = 1
    L_20 = 2
    L_21 = 3
    L_22 = 4
    L_24 = 5
    L_30 = 6
    L_31 = 7
    L_4 = 8
    L_5 = 9
    L_6 = 10
    L_7 = 11
    M_6 = 12
    M_7 = 13
    F_67 = 14

    # define spring delta L
    dlN = None
    dl4 = -q4

    x = [None] * 15
    x[0] = (L_b, -1, [L_1, M_6, M_7, F_67], 0, 0, 0, 0, False, False, False, dlN)  # Base

    x[1] = (L_1, L_b, [L_20], 0, _pi / 2, 0, q1 + _pi / 2, True, False, True, dlN)  # Yaw
    x[2] = (L_20, L_1, [L_21, L_31], 0, -_pi / 2, 0, q2 - _pi / 2, True, False, True, dlN)  # -- Intermediate
    x[3] = (L_21, L_20, [L_22], l_2L3, 0, 0, _pi / 2, False, False, False, dlN)  # Pitch Back
    x[4] = (L_22, L_21, [L_24, L_30], l_2H1, 0, 0, -q2 + _pi / 2, True, False, False,
            dlN)  # Pitch Front                                    )
    x[5] = (L_24, L_22, [L_30], l_2L2, 0, 0, q2, True, False, False, dlN)  # Pitch Bottom
    x[6] = (L_30, L_24, [L_4], l_3, -_pi / 2, q3 - l_RCC + l_2H1, 0, True, False, True, dlN)  # Pitch End
    x[7] = (L_31, L_20, [], l_2L3, -_pi / 2, q3, 0, True, False, False, dlN)  # Main Insertion
    x[8] = (L_4, L_30, [L_5], 0, 0, l_tool, q4, False, True, True, dl4)  # Intermediate Counterweight
    x[9] = (L_5, L_4, [L_6, L_7], 0, _pi / 2, 0, qmd5 + _pi / 2, False, True, True, dlN)  # Counterweight
    x[10] = (L_6, L_5, [], l_p2y, -_pi / 2, 0, qmd6 + _pi / 2, False, False, True, dlN)  # Tool Roll
    x[11] = (L_7, L_5, [], l_p2y, -_pi / 2, 0, qmd7 + _pi / 2, False, False, True, dlN)  # Tool Pitch

    x[12] = (M_6, L_b, [], 0, 0, 0, q6, False, True, True, dlN)  # Tool Yaw1 inert
    x[13] = (M_7, L_b, [], 0, 0, 0, q7, False, True, True, dlN)  # Tool Yaw2 inert
    x[14] = (F_67, L_b, [], 0, 0, 0, qmd7 - qmd6, False, False, True, dlN)  # q6 q7 coupled friction

    dh = x
    friction_type = ['Coulomb', 'viscous', 'offset']
    print("The generated model name is: ", model_name)
    print("The friction types include: ", friction_type)
    robot_def = RobotDef(dh, dh_convention='mdh', friction_type=friction_type)
    return robot_def



if __name__=="__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default='psm',
        help="model name",
    )
    parser.add_argument(
        '-p',
        '--long-print',
        action='store_true',
        help='specify if pretty print all parameters'
    )
    parser.add_argument(
        '-f',
        '--show-image',
        action='store_true',
        help='specify if show the images'
    )
    args = parser.parse_args()
    model_name = args.model_name
    flag_long_print = args.long_print
    flag_show_img = args.show_image

    ### create folder if necessary
    model_folder = os.path.join(dynamic_path, 'data', model_name, 'model')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        print('Created folder: ', model_folder)

    robot_def = create_robot_model(model_name)

    # M_motor2dvrk_q = np.array([[1.0186, 0, 0],
    #                            [-0.8306, 0.6089, 0.6089],
    #                            [0, -1.2177, 1.2177]])
    # M_model2dvrk_q = np.array([[1, 0, 0],
    #                            [0, 0.5, 0.5],
    #                            [0, -1, 1]])
    # M_motor2model_q = np.linalg.inv(M_model2dvrk_q) * M_motor2dvrk_q
    # M_motor2model_q[np.abs(M_motor2model_q) < 0.0001] = 0
    # print(M_motor2model_q)

    print('Robot joint type:')
    print(robot_def.coordinates_joint_type)

    if flag_long_print:
        print('Robot Parameters:')
        display(robot_def.bary_params)

    start_time = time.time()
    ### Generate the geometry based on robot description
    geom = Geometry(robot_def)
    print(f'Geometry construction time: {time.time() - start_time} s') # 598s

    if flag_show_img:
        angle = [0, 0, 0, 0, 0, 0, 0]
        print('Press q on the figure to Exit')
        geom.draw_geom(angle)
        plt.close('all')

    start_time = time.time()
    ### Create dynamics
    dyn = Dynamics(robot_def, geom)
    print(f'Dynamic construction time: {time.time() - start_time} s') # 11s

    if flag_long_print:
        display(sympy.Matrix(dyn.base_param))

    # Create the robot model
    robot_model = RobotModel(dyn)

    # save the model
    save_data(model_folder, model_name, robot_model)

    print('Saved {} parameters'.format(len(robot_model.base_param)))
