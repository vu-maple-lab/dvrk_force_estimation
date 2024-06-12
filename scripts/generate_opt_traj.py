import time
import numpy as np
from dvrk_dynamic_identification.utils import save_data, load_data
from dvrk_dynamic_identification.trajectory_optimization.traj_optimizer import TrajOptimizer
from dvrk_dynamic_identification.trajectory_optimization.traj_plotter import TrajPlotter
import tabulate
import os
import sys
dynamic_path = os.path.abspath(__file__+"/../../") # move to the parent folder [dvrk_force_estimation]
# print(dynamic_path)
sys.path.append(dynamic_path)
from IPython.display import display, HTML
import argparse

if __name__=="__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--trajectory-name",
        type=str,
        default='one',
        help="trajectory name",
    )
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
    trajectory_name = args.trajectory_name
    model_name = args.model_name
    flag_long_print = args.long_print
    flag_show_img = args.show_image

    model_folder = os.path.join(dynamic_path, 'data', model_name, 'model')

    robot_model = load_data(model_folder, model_name)

    trajectory_folder = os.path.join(dynamic_path, 'data', model_name, 'optimal_trajectory')

    base_freq = 0.18
    fourier_order = 6

    joint_constraints = []
    cartesian_constraints = []

    # M_motor2dvrk_q = np.array([[1.0186, 0, 0],
    #                            [-0.8306, 0.6089, 0.6089],
    #                            [0, -1.2177, 1.2177]])
    # M_model2dvrk_q = np.array([[1, 0, 0],
    #                            [0, 0.5, 0.5],
    #                            [0, -1, 1]])

    q_dvrk7 = 1.2177 * robot_model.coordinates[6] - 1.2177 * robot_model.coordinates[5]

    q_dvrk5 = 1.0186 * robot_model.coordinates[4]
    q_mod6 = -0.8306 * robot_model.coordinates[4] + 1.2178 * robot_model.coordinates[5]
    q_mod7 = -0.8306 * robot_model.coordinates[4] + 1.2178 * robot_model.coordinates[6]

    joint_constraints = [(robot_model.coordinates[0], -1.45, 1.45, -1.7, 1.7),
                         (robot_model.coordinates[1], -0.75, 0.8, -1.7, 1.7),
                         (robot_model.coordinates[2], 0.07, 0.235, -0.35, 0.35),
                         (robot_model.coordinates[3], -1.5, 1.5, -2, 2),
                         (q_dvrk5, -1.4, 1.4, -2, 2),
                         (q_dvrk7, 0.15, 3, -3, 3),
                         (q_mod7, -1.5, 1.5, -2, 2),
                         (q_mod6, -1.5, 1.5, -2, 2)]

    ### optimization
    start_time = time.time()
    traj_optimizer = TrajOptimizer(robot_model, fourier_order, base_freq,
                                   joint_constraints=joint_constraints,
                                   cartesian_constraints=cartesian_constraints)
    traj_optimizer.optimize()
    print(f'Trajectory Optimization time: {time.time() - start_time} s') # 288s

    reg_norm_mat = traj_optimizer.calc_normalize_mat()

    if flag_show_img:
        traj_optimizer.calc_frame_traj()
        traj_plotter = TrajPlotter(traj_optimizer.fourier_traj, traj_optimizer.frame_traj,
                                   traj_optimizer.const_frame_ind, robot_model.coordinates)
        traj_plotter.plot_desired_traj(traj_optimizer.x_result)
        ### if specify const_frame_ind, uncomment this line
        # traj_plotter.plot_frame_traj(True)

    # save the trajectory for later use
    dof_order_bf_x_norm = (traj_optimizer.fourier_traj.dof, fourier_order, base_freq, traj_optimizer.x_result,
                           reg_norm_mat)
    save_data(trajectory_folder, trajectory_name, dof_order_bf_x_norm)
    print(f'Save the trajectory {trajectory_name}.pkl in {trajectory_folder}')

    if flag_long_print:
        table = []
        table.append(["joint", 'qo'] +
                     ["a" + str(i + 1) for i in range(fourier_order)] +
                     ["b" + str(i + 1) for i in range(fourier_order)])
        for i in range(traj_optimizer.fourier_traj.dof):
            line = []
            line.append(robot_model.coordinates[i])
            line += np.round(traj_optimizer.x_result[i * (1 + fourier_order * 2): (i + 1) * (1 + fourier_order * 2)],
                             4).tolist()
            table.append(line)

        display(tabulate.tabulate(table))


