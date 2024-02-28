import sys
import torch
from network import *
import torch.nn as nn
import utils
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from dataset import indirectTrocarTestDataset

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
data = 'trocar'
contact = 'no_contact'
net = sys.argv[1]
epoch_to_use = 0#int(sys.argv[1])
seal = sys.argv[2]
is_seal = seal == 'seal'

JOINTS = utils.JOINTS
root = Path('../../simon_trocar_feb_27' )

def main():
    path = '../../simon_trocar_feb_27/test_contact'
    in_joints = [0,1,2,3,4,5]

    dataset = indirectTrocarTestDataset(path, utils.WINDOW, utils.SKIP, in_joints, seal=seal, net=net)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    preprocess = 'filtered_torque_trocar/1'

    model_root = []
    network_pred = np.array([])
    fs_pred_np = np.array([])
    torque_np = np.array([])
    for j in range(JOINTS):
        if is_seal:
            folder = "trocar" + str(j)
        else:
            folder = "trocar_no_cannula" + str(j)
        model_root.append(root / preprocess / folder)

    networks = []
    for j in range(JOINTS):
        networks.append(trocarNetwork(utils.WINDOW, len(in_joints)).to(device))
        utils.load_prev(networks[j], model_root[j], epoch_to_use)
        print("Loaded a " + str(j) + " model")

    all_pred = torch.tensor([])
    uncorrected_loss = 0
    corrected_loss = 0
    loss_fn = torch.nn.MSELoss()

    for i, (position, velocity, torque, jacobian, time, fs_pred) in enumerate(loader):
        position = position.to(device)
        velocity = velocity.to(device)
        torque = torque.to(device)
        fs_pred = fs_pred.to(device)

        step_loss = 0
        step_pred = time.unsqueeze(1)

        for j in range(JOINTS):
            posvel = torch.cat((position, velocity, fs_pred[:,[j]]), axis=1).contiguous()
            pred = networks[j](posvel) + fs_pred[:,[j]]
#            pred = networks[j](posvel)*utils.max_torque[j] + fs_pred[:,j].unsqueeze(1)
            loss = loss_fn(pred.squeeze(), torque[:,j])
            step_loss += loss.item()
            step_pred = torch.cat((step_pred, pred.detach().cpu()), axis=1)

            if j == 2:

                network_pred = np.append(network_pred, (networks[j](posvel)).cpu().detach().numpy())
                fs_pred_np = np.append(fs_pred_np, (fs_pred[:,[j]]).cpu().detach().numpy())
                torque_np = np.append(torque_np, (torque[:,[j]]).cpu().detach().numpy())

        corrected_loss += step_loss / 6
        in_joints = np.array(in_joints)
#        fs_pred = fs_pred[:,(in_joints+1)*utils.WINDOW-1]
        uncorrected_loss += loss_fn(fs_pred, torque)

        all_pred = torch.cat((all_pred, step_pred), axis=0) if all_pred.size() else step_pred

    print('Uncorrected loss: ', uncorrected_loss/len(loader))
    print('Corrected loss: ', corrected_loss/len(loader))
    np.savetxt('../../simon_trocar_feb_27/test_contact/' + 'filtered_corrected_torque_prediction.csv', all_pred.numpy())

    # After training, create a 2x3 grid of subplots for the losses
    fig, axs = plt.subplots(1, 1)  # 2 rows and 3 columns

    print(fs_pred_np.size)
    print(network_pred.size)
    print(torque_np.size)

    axs.plot(np.arange(1, len(fs_pred_np) + 1), network_pred, label='network_pred')
    axs.plot(np.arange(1, len(fs_pred_np) + 1), torque_np - fs_pred_np, label='diff')
    axs.set_title(f'Model {j + 1} Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')

    axs.legend()

    plt.tight_layout()  # Adjusts spacing between subplots
    plt.show()
    

if __name__ == "__main__":
    main()
