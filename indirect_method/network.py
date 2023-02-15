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

class trocarNetwork(nn.Module):
    def __init__(self, window, in_joints=6, out_joints=1):
        super(trocarNetwork, self).__init__()

        self.layer1 = nn.Linear(window * in_joints * 2 + 1, 256)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, out_joints)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        #        x = self.layer2(x)
        #        x = self.activation(x)
        x = self.layer3(x)
        #        x = self.tanh(x)
        return x


# Vaguely inspired by LSTM from https://github.com/BerkeleyAutomation/dvrkCalibration/blob/cec2b8096e3a891c4dcdb09b3161e2a407fee0ee/experiment/3_training/modeling/models.py
class torqueLstmNetwork(nn.Module):
    def __init__(self, batch_size, device, attn_nhead, joints=6, hidden_dim=256, num_layers=1):
        super(torqueLstmNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(joints * 2, hidden_dim, num_layers, batch_first=True)
        self.linear0 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_nhead)
        self.linear1 = nn.Linear(int(hidden_dim / 2), 1)

        self.linear3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.hidden = self.init_hidden(self.batch_size, self.device)

    def forward(self, x):
        # self.hidden = self.init_hidden(x.size()[0], self.device)
        x, self.hidden = self.lstm(x, self.hidden)
        #        x, self.hidden = self.lstm(x, self.hidden)
        #        self.hidden = tuple(state.detach() for state in self.hidden)

        # print(self.hidden[0].size())
        # print(x.size())

        # x = rearrange(x, 'bs sl fe -> sl bs fe')
        # x, _ = self.attn(query=x, key=self.hidden[0], value=self.hidden[0])
        # x = rearrange(x, 'bs sl fe -> sl bs fe')

        # x = self.linear3(x)

        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.tanh(x)
        return x

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device))


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


# Attention only for decoder
class LSTM_ATTN_Encoder_Only(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_attn_layers, attn_nhead, attn_hidden_dim,
                 output_size, device, rt_test, torch_attn=True):

        super(LSTM_ATTN_Encoder_Only, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # LSTM
        self.encoder = LSTM_Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers,
                                    rt_test=rt_test, device=device)

        # ATTN
        if torch_attn:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_nhead)
        else:
            self.attn = Decoder(num_layers=num_attn_layers, interm_dim=attn_hidden_dim, feat_dim=hidden_size,
                                nhead=attn_nhead, dropout=0)

        # FCN
        self.feedforward = nn.ModuleDict({
            'linear': nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, output_size)
            )
        })

        # Activation
        self.activation = nn.ReLU()

    def forward(self, input_tensor, hidden):
        # TODO:
        #  1.   embed input position and velocity to hidden state dimension
        #  2.   train one LSTM encoder on the embedded sequence, take the encoded hidden state and output
        #  3.   concatenate input with LSTM encoder hidden dimension, apply multi-head self-attention on combined sequence
        #  4.   attended output will be fed into LSTM decoder to generate output

        # encode
        encoder_output, encoder_hidden = self.encoder(input_tensor, hidden)
        encoder_output = rearrange(encoder_output, 'bs sl fe -> sl bs fe')

        # attend
        # attn_mask = generate_attn_mask(bs, ls)
        attn_output, _ = self.attn(query=encoder_output, key=encoder_hidden[0], value=encoder_hidden[0])

        # decode
        attn_output = rearrange(attn_output, 'sl bs fe -> bs sl fe')

        x = self.feedforward['linear'](attn_output)

        return x, encoder_hidden


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device, rt_test, num_layers=1):
        super(LSTM_Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.rt_test = rt_test

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def forward(self, x_input, hidden):
        bs, ls = x_input.shape[0], x_input.shape[1]

        if hidden is None:
            hidden = self.init_hidden(bs, self.device)

        hidden_states = hidden if self.rt_test else self.init_hidden(bs, self.device)
        lstm_out, hidden_states = self.lstm(x_input, hidden_states)

        return lstm_out, hidden_states


class Decoder(nn.Module):
    """
    Transformer Decoder:
    Stacked collections of decoder layers
    """

    def __init__(self, num_layers, interm_dim, feat_dim, nhead, dropout):
        super(Decoder, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model=feat_dim, dropout=dropout, max_len=5000)
        self.decoder_layer = DecoderLayer(interm_dim=interm_dim, feat_dim=feat_dim, nhead=nhead, dropout=dropout)
        self.decoders = get_clones(self.decoder_layer, num_layers)

    def forward(self, x):
        """
        :param x: position and velocity sequence
        """
        x = self.pos_encoder.forward(x)

        for module in self.decoders:
            x = module(x)

        return x


class DecoderLayer(nn.Module):
    """
    Force Estimation Decoder Layer:
    Applies self-attention on raw position and velocity sequence
    """

    def __init__(self, feat_dim, interm_dim, nhead, dropout):
        super(DecoderLayer, self).__init__()

        self.attn_layer = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)
        self.feedforward = FeedForwardLayer(feat_dim=feat_dim, interm_dim=interm_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: position and velocity sequence
        """
        # self-attn
        x = self.dropout(self.attn_layer(q=x, k=x, v=x, mask=None)) + x

        # feed forward
        x = self.feedforward(x)

        return x


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def generate_attn_mask(batch_size, length):
    """
    generate self-attention mask
    :param length: length of input kinematics sequence
    :return: upper triangular mask indices
    """
    mask = np.triu(np.ones((batch_size, length, length)), k=1).astype(np.bool_)
    mask_indices = torch.from_numpy(mask) == 0
    return mask_indices


def attention(q, k, v, d_k, mask=None, dropout=None):
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # apply mask if needed
    if mask is not None:
        mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -1e9)
    # apply dropout if needed
    score = F.softmax(score, dim=-1)
    if dropout is not None:
        score = dropout(score)
    # return attention
    attn = torch.matmul(score, v)
    return attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        if torch.equal(q, k) and torch.equal(k, v):
            assert q.shape[1] == k.shape[1] and k.shape[1] == v.shape[
                1], "self-attention input doesn't have equal length"
        num_batch = q.shape[0]
        _q = self.w_q(q).view(num_batch, -1, self.nhead, self.d_k).transpose(1, 2)
        _k = self.w_k(k).view(num_batch, -1, self.nhead, self.d_k).transpose(1, 2)
        _v = self.w_v(v).view(num_batch, -1, self.nhead, self.d_k).transpose(1, 2)

        attn = attention(_q, _k, _v, self.d_k, mask=mask, dropout=self.dropout)
        attn = attn.transpose(1, 2).contiguous().view(num_batch, -1, self.d_model)
        attn = self.linear(attn)

        return attn


class FeedForwardLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(feat_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, feat_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.linear2(self.activation(self.linear1(x)))) + x
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, kin):
        x = kin + self.pe[:kin.size(0), :]
        return self.dropout(x)

