from os.path import join
import tqdm
import torch
import torch.nn as nn
from dataset import *
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

max_torque = [20.0000, 20.0000, 20.0000, 0.2000, 0.6000, 0.6000]

JOINTS = 6
WINDOW = 30
SKIP = 1

def nrmse_loss(y_hat, y, j=0, verbose=False):
    if verbose:
        print(y.max(axis=0).values, y.min(axis=0).values)
        print(y_hat.max(axis=0).values, y_hat.min(axis=0).values)
    denominator = y.max(axis=0).values-y.min(axis=0).values
    rmse = torch.sqrt(torch.mean((y_hat-y)**2, axis=0))
    nrmse = rmse/denominator #range_torque[j] #
    
    return torch.mean(nrmse) * 100

def relELoss(y_hat, y):
    nominator = torch.sum((y_hat-y)**2)
    demonimator = torch.sum(y**2)
    error = torch.sqrt(nominator/denominator)
    return error * 100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_prev(network, model_root, epoch, optimizer=None, scheduler=None):
    if epoch == 0:
        model_path = model_root / 'model_joint_best.pt'
    else:
        model_path = model_root / 'model_joint_{}.pt'.format(epoch)
        
    if model_path.exists():
        state = torch.load(str(model_path))
        network.load_state_dict(state['model'])
        epoch = state['epoch'] + 1
        network.eval()

        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            network.train()
        print('Restored model, epoch {}'.format(epoch))
    else:
        print('Failed to restore model ' + str(model_path))
        exit()
        
    return epoch
        
def calculate_force(jacobian, joints):
    jacobian = jacobian.numpy()
    force = torch.zeros((joints.shape[0], 6))
    for i in range(joints.shape[0]):
        j = jacobian[i,:].reshape(6,6)
        jacobian_inv_t = torch.from_numpy(np.linalg.inv(j).transpose())
        force[i,:] = torch.matmul(jacobian_inv_t, joints[i,:])
    return force
        
save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))

