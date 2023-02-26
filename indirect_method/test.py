import sys
import torch
from network import *
import torch.nn as nn
import utils
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import indirectTestDataset, indirectDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
contact = 'with_contact'
data = 'free_space'

JOINTS = utils.JOINTS
epoch_to_use = 0  # int(sys.argv[1])
exp = sys.argv[1]  # sys.argv[2]
net = sys.argv[2]
seal = sys.argv[3]
preprocess = 'filtered_torque_si'  # sys.argv[4]
is_rnn = net == 'lstm'
if is_rnn:
    batch_size = 1
else:
    batch_size = 8192
root = Path('../..')

if seal == 'seal':
    fs = 'free_space'
elif seal == 'base':
    fs = 'no_cannula'

max_torque = torch.tensor(utils.max_torque).to(device)
print('device is: ', device)

ATTN_nhead=1

def main():
    all_pred = None
    if exp == 'train':
        path = '../../data_2_23/csv_si/train/' + data + '/'
    elif exp == 'val':
        path = '../../data_2_23/csv_si/val/' + data + '/'
    elif exp == 'test':
        # path = '../../csv_si/test/' + data + '/no_contact/'
        path = '../../data_2_23/csv_si/test/' + data + '/'
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
        model_root.append(root / preprocess / net / folder)

    networks = []
    for j in range(JOINTS):
        if is_rnn:
            networks.append(torqueTransNetwork(device, attn_nhead=ATTN_nhead).to(device))
            # networks.append(torqueLstmNetwork(batch_size, device, attn_nhead=ATTN_nhead).to(device))
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
