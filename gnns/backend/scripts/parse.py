import numpy as np
import collections
import json
import time
import argparse


def load_json(filepath):
    with open(str(filepath)) as json_file:
        json_load = json.load(json_file)
    return json_load


def str_to_func(pkg, func_name):
    return getattr(pkg, func_name)

# Return layer_list: all layers in sequence, param: a nested dict that {layer:{para: value}}


def json_to_dict(json_load):
    layer_list = []
    detail = {}
    param_dict = {}
    # Loop for layers
    for l in json_load['layers']:
        layer_list.append(l['type'])
        if l['parameters']:
            for p in l['parameters']:
                if 'activation' in p.keys() and p['activation']:
                    p['activation'] = str_to_func(F, p['activation'].lower())
                    detail.update(p)
                else:
                    detail.update(p)
                param_dict[l['type']] = detail
        else:
            param_dict[l['type']] = None
        detail = {}
    # Loop for layers ends
    optimizer_dict = json_load['optimizer']
    hyperparam_dict = json_load['hyperparameters']
    loss_dict = json_load['lossfunction']
    return layer_list, param_dict, optimizer_dict, hyperparam_dict, loss_dict


def get_layer_dict(layer_list):
    layer_dict = collections.OrderedDict()
    dglnn = dir(dgl.nn)
    trnn = dir(tr.nn)
    for l in layer_list:
        for f, g in zip(dglnn, trnn):
            if str(f) in l:
                layer_dict[l] = str(f)
            elif str(g) in l:
                layer_dict[l] = str(g)
    return layer_dict


def get_layer_list(layer_dict, param_dict):
    layer_list = []
    for x, y in layer_dict.items():
        try:
            layer = str_to_func(dgl.nn, y)
        except:
            layer = str_to_func(tr.nn, y)
        if param_dict[x]:
            layer_list.append(layer(**param_dict[x]))
        else:
            layer_list.append(layer())
    return layer_list


def Parser(json_path):
    layers = load_json(json_path)
    layer_list, param_dict, optimizer_dict, hyperparam_dict, loss_dict = json_to_dict(
        layers)
    layer_dict = get_layer_dict(layer_list)
    layer_list = get_layer_list(layer_dict, param_dict)
    return layer_list, optimizer_dict, hyperparam_dict, loss_dict
