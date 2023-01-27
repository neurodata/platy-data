from os.path import dirname, join
from pathlib import Path
import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt
from graspologic.utils import is_fully_connected

DATA_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = DATA_PATH / "docs" / "outputs"


def _get_folder(path=None):
    if path == None:
        path = DATA_PATH
    folder = path
    return folder


def load_all_annotations():
    dir = DATA_PATH / "all_annotations.csv"
    annotations = pd.read_csv(dir)
    return annotations


def load_connectome_annotations():
    dir = DATA_PATH / "connec_annotations.csv"
    connec_annots = pd.read_csv(dir)
    return connec_annots


def load_connectome_lcc_annotations():
    dir = DATA_PATH / "connec_lcc_annotations.csv"
    connec_lcc_annots = pd.read_csv(dir)
    return connec_lcc_annots


def load_connectome_normal_lcc_annotations():
    dir = DATA_PATH / "connec_lcc_normal_annotations.csv"
    connec_lcc_normal_annots = pd.read_csv(dir)
    return connec_lcc_normal_annots


def load_connectome_adj():
    dir = DATA_PATH / "adj_connectome.csv"
    adj_connec = pd.read_csv(dir)
    adj_connec = adj_connec.set_axis(list(adj_connec), axis="index")
    return adj_connec


def load_connectome_lcc_adj():
    dir = DATA_PATH / "adj_connectome_lcc.csv"
    adj_connec_lcc = pd.read_csv(dir)
    adj_connec_lcc = adj_connec_lcc.set_axis(list(adj_connec_lcc), axis="index")
    return adj_connec_lcc


def load_connectome_lcc_normal_adj():
    dir = DATA_PATH / "adj_connectome_normal_lcc.csv"
    adj_connec_normal_lcc = pd.read_csv(dir)
    adj_connec_normal_lcc = adj_connec_normal_lcc.set_axis(
        list(adj_connec_normal_lcc), axis="index"
    )
    return adj_connec_normal_lcc


def load_full_adj():
    dir = DATA_PATH / "full_adj.csv"
    full_adj = pd.read_csv(dir)
    full_adj = full_adj.set_axis(list(full_adj), axis="index")
    return full_adj


def load_weird_annotations():
    dir = DATA_PATH / "weird_annotations.csv"
    weird_annotations = pd.read_csv(dir)
    return weird_annotations


def load_left_adj():
    dir1 = DATA_PATH / "adj_left_normal_lcc.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    return left_adj


def load_right_adj():
    dir1 = DATA_PATH / "adj_right_normal_lcc.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    return right_adj


def load_head_adj():
    dir1 = DATA_PATH / "adj_head_normal_lcc.csv"
    head_adj = pd.read_csv(dir1)
    head_adj = head_adj.set_axis(list(head_adj), axis="index")

    return head_adj


def load_pygidium_adj():
    dir1 = DATA_PATH / "adj_pygidium_normal_lcc.csv"
    pyg_adj = pd.read_csv(dir1)
    pyg_adj = pyg_adj.set_axis(list(pyg_adj), axis="index")

    return pyg_adj


def load_0_adj():
    dir1 = DATA_PATH / "adj_0_normal_lcc.csv"
    adj_0 = pd.read_csv(dir1)
    adj_0 = adj_0.set_axis(list(adj_0), axis="index")

    return adj_0


def load_1_adj():
    dir1 = DATA_PATH / "adj_1_normal_lcc.csv"
    adj_1 = pd.read_csv(dir1)
    adj_1 = adj_1.set_axis(list(adj_1), axis="index")

    return adj_1


def load_2_adj():
    dir1 = DATA_PATH / "adj_2_normal_lcc.csv"
    adj_2 = pd.read_csv(dir1)
    adj_2 = adj_2.set_axis(list(adj_2), axis="index")

    return adj_2


def load_3_adj():
    dir1 = DATA_PATH / "adj_3_normal_lcc.csv"
    adj_3 = pd.read_csv(dir1)
    adj_3 = adj_3.set_axis(list(adj_3), axis="index")

    return adj_3


adj = load_right_adj()
print(adj)
