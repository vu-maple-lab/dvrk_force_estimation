import torch
import torch.nn as nn

import numpy as np
import math
import copy
from torch.nn import functional as F


# Network maps joint position and velocity to torque
class fsNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(fsNetwork, self).__init__()

        self.layer1 = nn.Linear(window * in_joints * 2, 256)
        # self.layer1 = nn.Linear(window * in_joints, 256) # TODO when removing verlocity
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, out_joints)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

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
        x = self.tanh(x)
        return x


# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueLstmNetwork(nn.Module):
    def __init__(self, batch_size, device, joints=6, hidden_dim=128, num_layers=1, is_train=False):
        super(torqueLstmNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(joints * 2, hidden_dim, num_layers, batch_first=True)
        self.linear0 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.linear1 = nn.Linear(int(hidden_dim / 2), 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.hidden = self.init_hidden(self.batch_size, self.device)
        self.is_train = is_train

    def forward(self, x):
        if self.is_train:
            self.hidden = self.init_hidden(x.size()[0], self.device)
        x, self.hidden = self.lstm(x, self.hidden)

        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.tanh(x)
        return x

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device))


# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueTransNetwork(nn.Module):
    def __init__(self, batch_size, device, attn_nhead, joints=6, hidden_dim=128, num_layers=1, dropout=0.00, is_train=False):
        super(torqueTransNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.lineark = nn.Linear(joints * 2, int(hidden_dim/2))
        self.linearq = nn.Linear(joints * 2, int(hidden_dim/2))
        self.linearv = nn.Linear(joints * 2, int(hidden_dim/2))
        self.attn = nn.MultiheadAttention(embed_dim=int(hidden_dim/2), num_heads=attn_nhead, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(joints * 2, int(hidden_dim/2))
        self.linear2 = nn.Linear(hidden_dim, joints * 2)
        self.linear3 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear4 = nn.Linear(int(hidden_dim/2), 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.layer_norm1 = nn.LayerNorm(int(hidden_dim/2))
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(int(hidden_dim/2), hidden_dim, num_layers, batch_first=True)
        self.hidden = self.init_hidden(self.batch_size, self.device)
        self.is_train = is_train

    def forward(self, x):
        k = self.lineark(x)
        q = self.linearq(x)
        v = self.linearv(x)

        k = self.relu(k)
        q = self.relu(q)
        v = self.relu(v)
        attn_output, _ = self.attn(query=q, key=k, value=v)

        # x = self.linear1(x)
        # x = self.relu(x)
        #
        # x = x + self.dropout1(attn_output)
        # x = self.layer_norm1(x)

        if self.is_train:
            self.hidden = self.init_hidden(attn_output.size()[0], self.device)
        x, self.hidden = self.lstm(attn_output, self.hidden)

        # x = self.linear1(x)
        # x = self.relu(x)
        #
        # x = x + self.dropout1(attn_output)
        # x = self.layer_norm1(x)

        # x = self.relu(x)
        #
        # # Feed-forward network
        # ff_output = self.feed_forward(x)
        # x = x + self.dropout2(ff_output)
        # x = self.layer_norm2(x)
        # x = self.relu(x)

        x = self.linear3(x)
        x = self.relu(x)

        x = self.linear4(x)
        x = self.tanh(x)

        return x

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device))



# class TransformerBlock(nn.Module):
#     def __init__(self, hidden_dim, num_heads, dropout=0.05):
#         super().__init__()
#
#         self.lineark = nn.Linear(hidden_dim, 2*hidden_dim)
#         self.linearq = nn.Linear(hidden_dim, 2*hidden_dim)
#         self.linearv = nn.Linear(hidden_dim, 2*hidden_dim)
#
#         self.linear1 = nn.Linear(hidden_dim, 2*hidden_dim)
#         self.linear2 = nn.Linear(2 * hidden_dim, hidden_dim)
#
#         self.relu = nn.ReLU()
#
#         self.self_attn = nn.MultiheadAttention(2*hidden_dim, num_heads, dropout=dropout)
#         self.layer_norm1 = nn.LayerNorm(2*hidden_dim)
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(2*hidden_dim,  4*hidden_dim),
#             nn.GELU(),
#             nn.Linear(4*hidden_dim, hidden_dim),
#         )
#         self.layer_norm2 = nn.LayerNorm(hidden_dim)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x):
#         k = self.lineark(x)
#         q = self.linearq(x)
#         v = self.linearv(x)
#
#         k = self.relu(k)
#         q = self.relu(q)
#         v = self.relu(v)
#
#
#         # Multi-head self-attention
#         attn_output, _ = self.self_attn(query=q, key=k, value=v)
#         x = self.linear1(x) + self.dropout1(attn_output)
#         x = self.layer_norm1(x)
#
#         # Feed-forward network
#         ff_output = self.feed_forward(x)
#         x = self.linear2(x) + self.dropout2(ff_output)
#         x = self.layer_norm2(x)
#
#         return x
#
# class torqueTransNetwork(nn.Module):
#     def __init__(self, device, attn_nhead, joints=6, hidden_dim=24, num_layers=1, dropout=0.05):
#         super().__init__()
#
#         self.device = device
#         self.linear1 = nn.Linear(joints * 2, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#
#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(hidden_dim, attn_nhead, dropout)
#             for _ in range(num_layers)
#         ])
#
#         self.linear2 = nn.Linear(hidden_dim, 1)
#         self.tanh = nn.Tanhshrink()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#
#         for transformer in self.transformer_blocks:
#             x = transformer(x)
#
#         x = self.linear2(x)
#         x = self.tanh(x)
#
#         return x


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

        return x