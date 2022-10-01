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


def load_left_adj_labels():
    dir1 = DATA_PATH / "adj_left.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemi_classes.csv"
    hemi_labels = pd.read_csv(dir2)
    left_labels = hemi_labels["l"]
    left_labels = [x for x in left_labels if str(x) != "nan"]

    return left_adj, left_labels


def load_right_adj_labels():
    dir1 = DATA_PATH / "adj_right.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemi_classes.csv"
    hemi_labels = pd.read_csv(dir2)
    right_labels = hemi_labels["r"]
    right_labels = [x for x in right_labels if str(x) != "nan"]

    return right_adj, right_labels


def load_head_adj_labels():
    dir1 = DATA_PATH / "adj_head.csv"
    head_adj = pd.read_csv(dir1)
    head_adj = head_adj.set_axis(list(head_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    head_labels = segs_labels["head"]
    head_labels = [x for x in head_labels if str(x) != "nan"]

    return head_adj, head_labels


def load_pygidium_adj_labels():
    dir1 = DATA_PATH / "adj_pygidium.csv"
    pyg_adj = pd.read_csv(dir1)
    pyg_adj = pyg_adj.set_axis(list(pyg_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    pyg_labels = segs_labels["pygidium"]
    pyg_labels = [x for x in pyg_labels if str(x) != "nan"]

    return pyg_adj, pyg_labels


def load_0_adj_labels():
    dir1 = DATA_PATH / "adj_0.csv"
    adj_0 = pd.read_csv(dir1)
    adj_0 = adj_0.set_axis(list(adj_0), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_0 = segs_labels["0"]
    labels_0 = [x for x in labels_0 if str(x) != "nan"]

    return adj_0, labels_0


def load_1_adj_labels():
    dir1 = DATA_PATH / "adj_1.csv"
    adj_1 = pd.read_csv(dir1)
    adj_1 = adj_1.set_axis(list(adj_1), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_1 = segs_labels["1"]
    labels_1 = [x for x in labels_1 if str(x) != "nan"]

    return adj_1, labels_1


def load_2_adj_labels():
    dir1 = DATA_PATH / "adj_2.csv"
    adj_2 = pd.read_csv(dir1)
    adj_2 = adj_2.set_axis(list(adj_2), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_2 = segs_labels["2"]
    labels_2 = [x for x in labels_2 if str(x) != "nan"]

    return adj_2, labels_2


def load_3_adj_labels():
    dir1 = DATA_PATH / "adj_3.csv"
    adj_3 = pd.read_csv(dir1)
    adj_3 = adj_3.set_axis(list(adj_3), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_3 = segs_labels["3"]
    labels_3 = [x for x in labels_3 if str(x) != "nan"]

    return adj_3, labels_3


left_adj, left_labels = load_left_adj_labels()
print(left_adj)
