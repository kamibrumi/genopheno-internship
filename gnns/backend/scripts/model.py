import dgl
import dgl.nn
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import itertools

from parser import Parser
layer_list, optimizer_dict, hyperparam_dict, loss_dict = Parser(
    '/content/drive/MyDrive/VALT/gnns/backend/layers_example.json')


class Model(nn.Module):
    def __init__(self, g):
        super(Model, self).__init__()
        self.g = g
        self.layers = nn.ModuleList(layer_list)

    def forward(self, h):
        for layer in self.layers:
            if 'Dropout' in str(layer):
                h = layer(h)
            else:
                h = layer(self.g, h)

        return h
