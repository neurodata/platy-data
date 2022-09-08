from os.path import dirname, join
from pathlib import Path
import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt

DATA_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = DATA_PATH / "docs" / "outputs"


def _get_folder(path=None):
    if path == None:
        path = DATA_PATH
    folder = path
    return folder


def load_annotations():
    dir = DATA_PATH / "annotations.csv"
    annotations = pd.read_csv(dir)
    return annotations


def load_connectome_adj():
    dir = DATA_PATH / "adj_connectome.csv"
    adj_connec = pd.read_csv(dir)
    adj_connec = adj_connec.set_axis(list(adj_connec), axis="index")
    return adj_connec


def load_full_adj():
    dir = DATA_PATH / "full_adj.csv"
    full_adj = pd.read_csv(dir)
    full_adj = full_adj.set_axis(list(full_adj), axis="index")
    return full_adj


print(load_annotations())