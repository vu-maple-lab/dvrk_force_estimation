import torch
import torch.nn as nn


# Network maps joint position and velocity to torque
class torqueNetwork(nn.Module):
    def __init__(self, window):
        super(torqueNetwork, self).__init__()

        self.layer1 = nn.Linear(window*12, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.layer4(x)
        return x

# Network maps joint position and velocity to torque
class trocarNetwork(nn.Module):
    def __init__(self, window):
        super(trocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window*12, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.layer4(x)
        return x
    

# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueLstmNetwork(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=100, num_layers=2):
        super(torqueLstmNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers)  
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden = self.init_hidden(x)
        out, _ = self.lstm(x, hidden)
        out = self.linear(out[-1])
        return out

    def init_hidden(self, x):
        if next(self.parameters()).is_cuda:
            return (torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float().cuda(),
                    torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float().cuda())
        return (torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float(),
                torch.zeros(self.num_layers , x.size(1), self.hidden_dim).float())

    
# Network to do direct velocity to force estimate
class forceNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(forceNetwork, self).__init__()

        self.layer1 = nn.Linear(in_channels, 120)
        self.layer2 = nn.Linear(120, 120)
        self.layer3 = nn.Linear(120, 120)
        self.layer4 = nn.Linear(120, 120)
        self.layer5 = nn.Linear(120, out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.layer5(x)
#        x = self.activation(x)
#        x = self.layer6(x)
#        x = self.activation(x)
#        x = self.layer7(x)
        
        return x
