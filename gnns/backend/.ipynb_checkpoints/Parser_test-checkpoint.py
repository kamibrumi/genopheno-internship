{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "DenseGraphConv\n",
      "SumPooling\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[DenseGraphConv(), SumPooling()]"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# equivalences between jsoin and python: https://www.geeksforgeeks.org/read-json-file-using-python/\n",
    "import json\n",
    "import dgl\n",
    "import numpy as np\n",
    "import torch as th\n",
    "from dgl.nn import DenseGraphConv\n",
    "from dgl.nn import SumPooling\n",
    "\n",
    "class Layers:\n",
    "    \n",
    "    def __init__(self):\n",
    "        pass\n",
    "    \n",
    "    def get_DenseGraphConv(self, parameters):\n",
    "        # TODO figure out how to pass the parameters. maybe this would help: https://realpython.com/python-kwargs-and-args/#passing-multiple-arguments-to-a-function\n",
    "        # https://www.geeksforgeeks.org/args-kwargs-python/ \n",
    "        #print(parameters)\n",
    "        return DenseGraphConv(**parameters) # Can use DenseGraphConv(**kwargs) for dict or use *args for tuples \n",
    "    \n",
    "    def get_SumPooling(self, parameters):\n",
    "        return SumPooling()\n",
    "\n",
    "class Parser:\n",
    "    def __init__(self, json_filename):\n",
    "        self.json_filename = json_filename\n",
    "        self.parameters = {} #Zeyu: Add an self.parameters to store params for different layers \n",
    "        \n",
    "    def parse_parameters(self, parameters): # Here parameters need to be parsed in parse()\n",
    "        \n",
    "        params = {}\n",
    "        for d in parameters:\n",
    "            params.update(d)\n",
    "        return params\n",
    "    \n",
    "    # the parse function parses the layers\n",
    "    def parse(self):\n",
    "        layers = [] # where we will store all the layer objects\n",
    "        \n",
    "        #parse json file\n",
    "        f = open(self.json_filename)\n",
    "        # returns JSON object as a dictionary\n",
    "        data = json.load(f)\n",
    "        n_layers = data[\"n_layers\"]\n",
    "        layers_list = data[\"layers\"]\n",
    "        print(len(layers_list))\n",
    "        \n",
    "        for layer in layers_list:\n",
    "            print(layer[\"type\"])\n",
    "            \n",
    "            # the following line all it does is to fetch the function Layers.get_<name of layer>()\n",
    "            layer_getter = getattr(Layers, \"get_\" + layer[\"type\"])\n",
    "            \n",
    "            # now we call the function Layers.get_<name of layer>() and we obtain a layer as a result\n",
    "            params = self.parse_parameters(layer[\"parameters\"])\n",
    "            layer = layer_getter(self, params)\n",
    "            \n",
    "            self.parameters[str(layer)] = params #Store params for each layer to self.parameters\n",
    "            \n",
    "            # append that layer to the list of layers\n",
    "            layers.append(layer)\n",
    "        \n",
    "        #print(self.parameters)\n",
    "        \n",
    "        return layers\n",
    "\n",
    "p = Parser(\"layers_example.json\")\n",
    "layers = p.parse()\n",
    "layers\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'in_feats': 10,\n",
       " 'out_feats': 2,\n",
       " 'norm': 'both',\n",
       " 'bias': True,\n",
       " 'activation': None}"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Parser.parameters will look like this. We can pass this on to Layers.get_DenseGraphConv\n",
    "p.parameters['DenseGraphConv()']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DenseGraphConv()"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#This one shows **args works\n",
    "l = Layers()\n",
    "l.get_DenseGraphConv(p.parameters['DenseGraphConv()'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Original JSON:\n",
    "\n",
    "# {\n",
    "#   \"layers\": [\n",
    "#   {\n",
    "#       \"type\": \"DenseGraphConv\",\n",
    "#       \"parameters\": [\n",
    "#           {\"in_feats\": 10},\n",
    "#           {\"out_feats\": 2},\n",
    "#           {\"norm\": \"both\"},\n",
    "#           {\"bias\": true},\n",
    "#           {\"activation\": null}\n",
    "#       ]\n",
    "#   },\n",
    "#   {\n",
    "#       \"type\": \"SumPooling\",\n",
    "#       \"parameters\": []\n",
    "#   }\n",
    "#   ],\n",
    "#   \"n_layers\": 2\n",
    "# }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Some errors I was receiving\n",
    "- Somthing in the lines that you're missing _C, see: https://stackoverflow.com/questions/54408973/name-c-is-not-defined-pytorchjupyter-notebook, then restart the notebook\n",
    "- No module named 'urllib3'. Solution: conda install -c anaconda urllib3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
