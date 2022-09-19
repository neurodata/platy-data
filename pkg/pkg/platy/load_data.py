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


print(load_connectome_normal_lcc_annotations())
print(load_weird_annotations())
