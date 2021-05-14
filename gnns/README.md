# Documenting Anna's code

## M9Edge_predict_mu_alpha_.ipynb
1. QM9EdgeDataset: https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.QM9EdgeDataset with focus on just the 'mu' label_keys
2. Mini-batching made the runs quicker but less accurate depending on the learning rate. Batching is meant to be more efficient and accurate but looking at the MSE, batching did not make it more accurate. 
3. Output is the predictions per graph of the 'mu' value (a float). 
4. Yes this needs to be run on the cluster. Training the model took a lot of time and power even with batching and small epoch sizes.
5. Use MSE and MRE as the metric of accuracy
6. Just Used linear layer 

## Node Regression.ipynb
1. CiteseerGraphDataset: https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.CiteseerGraphDataset
2. Preferred mini-batching. 
3. Output is an int
4. No, it does not need to be run on the cluster
5. Use MSE and MRE as the metric of accuracy
6. Used SAGEConv
This regression has not be completed because we started exploring other datasets but you can use code from predict number of edges to get results (just need to change the dataset). 

## Predict_num_edges_minibatches.ipynb
1. MiniGCDataset:https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.MiniGCDataset
2. Used batching for this one but preferred no mini-batching. No batching gave better results
3. Output is a float (the number of edges predicted)
4. It does need to be run on the cluster. 
5. Use MSE and MRE as the metric of accuracy
6. Just Used linear layer 

## Sentiment Tree Regression v2.ipynb
1. SSTDataset:https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.SSTDataset
2. NA
3. Output is a float (the number of edges predicted)
4. It does need to be run on the cluster. 
5. Use MSE and MRE as the metric of accuracy
6. Using SAGEConv but not compatible with dataset and will try linear layer

Currently running into error because graph structure (containing labels per node) does not match the graph structures from datasets we have previously worked on. 

## Sentiment Tree Regression.ipynb
Experimental notebook, can delete

## networkx_color_nodes_by_attr.ipynb
Coloring nodes of based on node attributes.

## predict_num_edges_individual_graph.ipynb
1. MiniGCDataset:https://docs.dgl.ai/api/python/dgl.data.html#dgl.data.MiniGCDataset
2. Preferred no mini-batching; it gave better MSE and MRE results
3. Output is a float (the number of edges predicted)
4. It does need to be run on the cluster. 
5. Use MSE and MRE as the metric of accuracy
6. Just Used linear layer 

### Some notes on documentation: Please...
1. Add what the input/dataset you're using, if possible with a link to it.
2. Add for each code above if you used mini-batching or just traditional sampling with no mini-batching.
3. What is the output? label per node? label per graph? are the labels numerical or categorical?
4. Does this code need to be run in the cluster (does it take too much time to train locally?)
5. What performance metrics do you use for training and for testing?
6. Did you use an interesting layer other than the linear layer for any of these models?
