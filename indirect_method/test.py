import sys
import torch
from network import *
import torch.nn as nn
import utils
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import indirectTestDataset, indirectDataset

#mkdir -p ~/filtered_torque_colon_9_26/lstm/PSM1
# cd ~/dvrk_force_estimation/indirect_method
# python3 train.py free_space lstm PSM1

#python3 test.py test lstm seal PSM1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
contact = 'no_contact'
data = 'free_space'

JOINTS = utils.JOINTS
epoch_to_use = 0  # int(sys.argv[1])
exp = sys.argv[1]  # sys.argv[2]
net = sys.argv[2]
seal = sys.argv[3]

arm = sys.argv[4]

preprocess = 'filtered_torque_colon_9_26'  # sys.argv[4]
folder = net + '/' + arm + '/' + data
is_rnn = net != 'ff'
if is_rnn:
    batch_size = 1
else:
    batch_size = 8192
root = Path('..')

fs = 'no_cannula'
if seal == 'seal':
    fs = 'free_space'

max_torque = torch.tensor(utils.max_torque).to(device)
print('device is: ', device)

ATTN_nhead=1

# Coordinate transform
flip_y_axis  = np.diag([-1.0, -1.0, 1.0])
basis        = np.array([[ 0,  0, -1],
                          [ 0,  1,  0],
                          [ 1,  0,  0]], dtype=float)
angle_z      = np.deg2rad(-30)
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
    all_pred = None
    if exp == 'train':
        path = '../Data/train_csv/'
    elif exp == 'val':
        path = '../Data/val_csv/'
    elif exp == 'test':
        path = '../Data/test_csv/'
    else:
        path = '../../csv/test/' + data + '/' + contact + '/' + exp + '/'
        path = '../../csv/test/' + data + '/' + contact + '/' + exp + '/'
    in_joints = [0, 1, 2, 3, 4, 5]

    if is_rnn:
        window = 1000
    else:
        window = utils.WINDOW

    if is_rnn:
        dataset = indirectDataset(path, window, utils.SKIP, in_joints, is_rnn=is_rnn)
    else:
        dataset = indirectTestDataset(path, window, utils.SKIP, in_joints, is_rnn=is_rnn)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model_root = []
    for j in range(JOINTS):
        folder = fs + str(j)
        print("###### the name of loaded model is: #########", root / preprocess / net / folder)
        model_root.append(root / preprocess / net / arm / folder)

    networks = []
    for j in range(JOINTS):
        if net == 'lstm':
            networks.append(torqueLstmNetwork(batch_size, device).to(device))
        elif net == 'attn':
            networks.append(torqueTransNetwork(batch_size, device, attn_nhead=ATTN_nhead).to(device))
        else:
            networks.append(fsNetwork(window).to(device))


    for j in range(JOINTS):
        # print("###### the name of loaded model is: #########", root / preprocess / net / folder)
        utils.load_prev(networks[j], model_root[j], epoch_to_use)
        print("Loaded a " + str(j) + " model")

    loss_fn = torch.nn.MSELoss()
    all_loss = 0
    all_pred = torch.tensor([])
    all_time = torch.tensor([])

    # for i, (position, velocity, torque, time) in enumerate(loader):
    for i, (position, velocity, torque, jacobian, time) in enumerate(loader):
        position = position.to(device)
        velocity = velocity.to(device)
        if is_rnn:
            posvel = torch.cat((position, velocity), axis=2).contiguous()
        else:
            posvel = torch.cat((position, velocity), axis=1).contiguous()
            # posvel = position

        if is_rnn:
            time = time.permute((1, 0))
        torque = torque.squeeze()

        cur_pred = torch.zeros(torque.size())
        for j in range(JOINTS):
            hidden = None

            # pred, _ = networks[j](posvel, hidden)
            pred = networks[j](posvel)
            pred = pred.squeeze().detach()
            ##############################
            pred = pred * max_torque[j]
            ##############################
            cur_pred[:, j] = pred.cpu()

        loss = loss_fn(cur_pred, torque)
        all_loss += loss.item()

        if is_rnn:
            time = time.squeeze(-1)

        all_time = torch.cat((all_time, time.cpu()), axis=0) if all_time.size() else time.cpu()
        all_pred = torch.cat((all_pred, cur_pred.cpu()), axis=0) if all_pred.size() else cur_pred.cpu()

    all_pred = torch.cat((all_time.unsqueeze(1), all_pred), axis=1)
    print(path + net + '_' + seal + '_pred_' + preprocess + '.csv', all_pred.numpy())
    np.savetxt(path + net + '_' + seal + '_pred_' + preprocess + '.csv', all_pred.numpy())

    print('Loss: ', all_loss)


if __name__ == "__main__":
    main()