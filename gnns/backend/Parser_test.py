# equivalences between jsoin and python: https://www.geeksforgeeks.org/read-json-file-using-python/
import json
import dgl
import numpy as np
import torch as th
from dgl.nn import GraphConv
from dgl.nn import SumPooling

class Layers:
    
    def __init__(self):
        pass
    
    def get_GraphConv(self, parameters):
        # TODO figure out how to pass the parameters. maybe this would help: https://realpython.com/python-kwargs-and-args/#passing-multiple-arguments-to-a-function
        # https://www.geeksforgeeks.org/args-kwargs-python/ 
        #print(parameters)
        return GraphConv(**parameters) # Can use DenseGraphConv(**kwargs) for dict or use *args for tuples 
    
    def get_SumPooling(self, parameters):
        return SumPooling()

class Parser:
    def __init__(self, json_filename):
        self.json_filename = json_filename
        self.parameters = {} #Zeyu: Add an self.parameters to store params for different layers 
        
    def parse_parameters(self, parameters): # Here parameters need to be parsed in parse()
        
        params = {}
        for d in parameters:
            params.update(d)
        return params
    
    # the parse function parses the layers
    def parse(self):
        layers = [] # where we will store all the layer objects
        
        #parse json file
        f = open(self.json_filename)
        # returns JSON object as a dictionary
        data = json.load(f)
        n_layers = data["n_layers"]
        layers_list = data["layers"]
        
        for layer in layers_list:
            print(layer["type"])
            
            # the following line all it does is to fetch the function Layers.get_<name of layer>()
            #layer_getter = getattr(Layers, "get_" + layer["type"])
            layer_getter = getattr(Layers, "get_" + layer["type"])
            
            # now we call the function Layers.get_<name of layer>() and we obtain a layer as a result
            params = self.parse_parameters(layer["parameters"])
            layer = layer_getter(self, params)
            
            #Store params for each layer to self.parameters
            self.parameters[str(layer).split('(', 1)[0]] = params 
            
            # append that layer to the list of layers
            layers.append(layer)
        
        #print(self.parameters)
        
        return layers