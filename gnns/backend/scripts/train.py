from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import time
import collections
import dgl
import dgl.nn
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

from parser import *
from model import Model


def evaluate(model, features, labels, mask):
    model.eval()
    with tr.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = tr.max(logits, dim=1)
        correct = tr.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


############ Load Data (For test only. Should not in this file) ####################
data = CoraGraphDataset()
g = data[0]
features = g.ndata['feat']  # same as weights in GraphSAGE exmaple
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()

print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
      (n_edges, n_classes,
       train_mask.int().sum().item(),
       val_mask.int().sum().item(),
       test_mask.int().sum().item()))
####################################################################################


layer_list, optimizer_dict, hyperparam_dict, loss_dict = Parser(
    '/content/drive/MyDrive/VALT/gnns/backend/layers_example.json')

model = Model(g)

loss_fcn = str_to_func(tr.nn, loss_dict['type'])(
    **loss_dict['parameters'])  # Should be customized
optimizer = str_to_func(tr.optim, optimizer_dict['type'])
optimizer = optimizer(model.parameters(),  **optimizer_dict['parameters'])

# Set hyperparameters
epoch_range = range(hyperparam_dict['epochs'])
#batch_size = hyperparam_dict['batchs']

# history tracking
epoch_dict = collections.OrderedDict()
epoch_list = []
trloss_list = []
teloss_list = []

dur = []
for epoch in epoch_range:
    model.train()
    if epoch >= 3:
        t0 = time.time()

    # forward
    logits = model(features)
    loss = loss_fcn(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # backward
    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(model, features, labels, val_mask)
    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
          "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                         acc, n_edges / np.mean(dur) / 1000))
    epoch_list.append(epoch)
    trloss_list.append(loss.item())

print()
acc = evaluate(model, features, labels, test_mask)
print("Test accuracy {:.2%}".format(acc))

epoch_dict['epochs'] = epoch_list
epoch_dict['train_loss'] = trloss_list
epoch_dict['test_loss'] = teloss_list
