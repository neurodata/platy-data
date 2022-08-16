import pymaid
import logging
import pandas as pd
import numpy as np
import networkx as nx
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt
from networkx import from_numpy_array
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.layouts import layout_tsne
from graspologic.plot.plot import networkplot
from graspologic.utils import is_fully_connected, largest_connected_component

rm = pymaid.CatmaidInstance(
    server="https://catmaid.jekelylab.ex.ac.uk/#",
    project_id=11,
    api_token=None,
    http_user=None,
    http_password=None,
)
logging.getLogger("pymaid").setLevel(logging.WARNING)
pymaid.clear_cache()

skids = pymaid.get_skids_by_annotation("connectome")
adj = pymaid.adjacency_matrix(skids)
adj = adj.to_numpy()
lcc_adj = largest_connected_component(adj)
networkx_adj = from_numpy_array(lcc_adj)

X, node_pos = layout_tsne(networkx_adj, perplexity=50, n_iter=10000)
x_pos = []
y_pos = []
node_comms = []
for pos in node_pos:
    node_comms.append(pos[4])
    x_pos.append(pos[1])
    y_pos.append(pos[2])

node_comms = np.array(node_comms)
x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
degrees = np.sum(lcc_adj, axis=0)

plot = networkplot(
    adjacency=lcc_adj,
    x=x_pos,
    y=y_pos,
    node_hue=node_comms,
    palette="tab10",
    node_size=degrees,
    node_sizes=(20, 200),
    edge_hue="source",
    edge_alpha=0.5,
    edge_linewidth=0.5,
)
plt.show()
