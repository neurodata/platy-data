import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from giskard.plot import adjplot, scattermap
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.plot import networkplot
from matplotlib.patheffects import Normal, Stroke

"""
from pkg.data import (
    load_maggot_graph,
    load_network_palette,
    load_node_palette,
    load_unmatched,
)
"""
# from pkg.io import get_environment_variables
# from pkg.io import glue as default_glue
# from pkg.io import savefig
# from pkg.plot import set_theme
from pkg.platy import (
    load_0_adj_labels,
    load_1_adj_labels,
    load_2_adj_labels,
    load_3_adj_labels,
    load_head_adj_labels,
    load_pygidium_adj_labels,
    load_left_adj_labels,
    load_right_adj_labels,
)
from scipy.cluster import hierarchy
from umap import UMAP


# load the adjs
