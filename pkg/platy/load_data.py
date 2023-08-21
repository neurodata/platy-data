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

date = "7_24_23"
DATA_PATH = Path(__file__).parent.parent.parent
# DATA_PATH = DATA_PATH / "docs" / "outputs"
DATA_PATH = DATA_PATH / "docs" / "outputs" / f"{date}"


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


def load_connectome_lcc_annotations_v2():
    dir = DATA_PATH / "connec_lcc_annotations_v2.csv"
    connec_lcc_annots = pd.read_csv(dir)
    connec_lcc_annots.pop(connec_lcc_annots.columns[0])
    return connec_lcc_annots


def load_connectome_normal_lcc_annotations():
    dir = DATA_PATH / "connec_lcc_normal_annotations.csv"
    connec_lcc_normal_annots = pd.read_csv(dir)
    return connec_lcc_normal_annots


def load_connectome_normal_lcc_annotations_v2():
    dir = DATA_PATH / "connec_lcc_normal_annotations_v2.csv"
    connec_lcc_normal_annots = pd.read_csv(dir)
    connec_lcc_normal_annots.pop(connec_lcc_normal_annots.columns[0])
    return connec_lcc_normal_annots


def load_connectome_normal_lcc_annotations_v3():
    dir = DATA_PATH / "connec_lcc_normal_annotations_v3.csv"
    connec_lcc_normal_annots = pd.read_csv(dir)
    connec_lcc_normal_annots.pop(connec_lcc_normal_annots.columns[0])
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


def load_left_adj_orig():
    dir1 = DATA_PATH / "adj_left.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    return left_adj


def load_right_adj_orig():
    dir1 = DATA_PATH / "adj_right.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    return right_adj


def load_head_adj_orig():
    dir1 = DATA_PATH / "adj_head.csv"
    head_adj = pd.read_csv(dir1)
    head_adj = head_adj.set_axis(list(head_adj), axis="index")

    return head_adj


def load_pygidium_adj_orig():
    dir1 = DATA_PATH / "adj_pygidium.csv"
    pyg_adj = pd.read_csv(dir1)
    pyg_adj = pyg_adj.set_axis(list(pyg_adj), axis="index")

    return pyg_adj


def load_0_adj_orig():
    dir1 = DATA_PATH / "adj_0.csv"
    adj_0 = pd.read_csv(dir1)
    adj_0 = adj_0.set_axis(list(adj_0), axis="index")

    return adj_0


def load_1_adj_orig():
    dir1 = DATA_PATH / "adj_1.csv"
    adj_1 = pd.read_csv(dir1)
    adj_1 = adj_1.set_axis(list(adj_1), axis="index")

    return adj_1


def load_2_adj_orig():
    dir1 = DATA_PATH / "adj_2.csv"
    adj_2 = pd.read_csv(dir1)
    adj_2 = adj_2.set_axis(list(adj_2), axis="index")

    return adj_2


def load_3_adj_orig():
    dir1 = DATA_PATH / "adj_3.csv"
    adj_3 = pd.read_csv(dir1)
    adj_3 = adj_3.set_axis(list(adj_3), axis="index")

    return adj_3


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


def load_left_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_left_with_class.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemis_classes.csv"
    hemi_labels = pd.read_csv(dir2)
    left_labels = hemi_labels["l"]
    left_labels = [x for x in left_labels if str(x) != "nan"]

    return left_adj, left_labels


def load_right_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_right_with_class.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemis_classes.csv"
    hemi_labels = pd.read_csv(dir2)
    right_labels = hemi_labels["r"]
    right_labels = [x for x in right_labels if str(x) != "nan"]

    return right_adj, right_labels


def load_head_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_head_with_class.csv"
    head_adj = pd.read_csv(dir1)
    head_adj = head_adj.set_axis(list(head_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    head_labels = segs_labels["head"]
    head_labels = [x for x in head_labels if str(x) != "nan"]

    return head_adj, head_labels


def load_pygidium_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_pygidium_with_class.csv"
    pyg_adj = pd.read_csv(dir1)
    pyg_adj = pyg_adj.set_axis(list(pyg_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    pyg_labels = segs_labels["pygidium"]
    pyg_labels = [x for x in pyg_labels if str(x) != "nan"]

    return pyg_adj, pyg_labels


def load_0_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_0_with_class.csv"
    adj_0 = pd.read_csv(dir1)
    adj_0 = adj_0.set_axis(list(adj_0), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_0 = segs_labels["0"]
    labels_0 = [x for x in labels_0 if str(x) != "nan"]

    return adj_0, labels_0


def load_1_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_1_with_class.csv"
    adj_1 = pd.read_csv(dir1)
    adj_1 = adj_1.set_axis(list(adj_1), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_1 = segs_labels["1"]
    labels_1 = [x for x in labels_1 if str(x) != "nan"]

    return adj_1, labels_1


def load_2_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_2_with_class.csv"
    adj_2 = pd.read_csv(dir1)
    adj_2 = adj_2.set_axis(list(adj_2), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_2 = segs_labels["2"]
    labels_2 = [x for x in labels_2 if str(x) != "nan"]

    return adj_2, labels_2


def load_3_adj_labels_with_class():
    dir1 = DATA_PATH / "adj_3_with_class.csv"
    adj_3 = pd.read_csv(dir1)
    adj_3 = adj_3.set_axis(list(adj_3), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes.csv"
    segs_labels = pd.read_csv(dir2)
    labels_3 = segs_labels["3"]
    labels_3 = [x for x in labels_3 if str(x) != "nan"]

    return adj_3, labels_3


def load_left_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_left_with_class_v2.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemis_classes_v2.csv"
    hemi_labels = pd.read_csv(dir2)
    left_labels = hemi_labels["left"]
    left_labels = [x for x in left_labels if str(x) != "nan"]

    return left_adj, left_labels


def load_right_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_right_with_class_v2.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemis_classes_v2.csv"
    hemi_labels = pd.read_csv(dir2)
    right_labels = hemi_labels["right"]
    right_labels = [x for x in right_labels if str(x) != "nan"]

    return right_adj, right_labels


def load_head_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_head_with_class_v2.csv"
    head_adj = pd.read_csv(dir1)
    head_adj = head_adj.set_axis(list(head_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v2.csv"
    segs_labels = pd.read_csv(dir2)
    head_labels = segs_labels["head"]
    head_labels = [x for x in head_labels if str(x) != "nan"]

    return head_adj, head_labels


def load_pygidium_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_pygidium_with_class_v2.csv"
    pyg_adj = pd.read_csv(dir1)
    pyg_adj = pyg_adj.set_axis(list(pyg_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v2.csv"
    segs_labels = pd.read_csv(dir2)
    pyg_labels = segs_labels["pygidium"]
    pyg_labels = [x for x in pyg_labels if str(x) != "nan"]

    return pyg_adj, pyg_labels


def load_0_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_0_with_class_v2.csv"
    adj_0 = pd.read_csv(dir1)
    adj_0 = adj_0.set_axis(list(adj_0), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v2.csv"
    segs_labels = pd.read_csv(dir2)
    labels_0 = segs_labels["0"]
    labels_0 = [x for x in labels_0 if str(x) != "nan"]

    return adj_0, labels_0


def load_1_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_1_with_class_v2.csv"
    adj_1 = pd.read_csv(dir1)
    adj_1 = adj_1.set_axis(list(adj_1), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v2.csv"
    segs_labels = pd.read_csv(dir2)
    labels_1 = segs_labels["1"]
    labels_1 = [x for x in labels_1 if str(x) != "nan"]

    return adj_1, labels_1


def load_2_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_2_with_class_v2.csv"
    adj_2 = pd.read_csv(dir1)
    adj_2 = adj_2.set_axis(list(adj_2), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v2.csv"
    segs_labels = pd.read_csv(dir2)
    labels_2 = segs_labels["2"]
    labels_2 = [x for x in labels_2 if str(x) != "nan"]

    return adj_2, labels_2


def load_3_adj_labels_with_class_v2():
    dir1 = DATA_PATH / "adj_3_with_class_v2.csv"
    adj_3 = pd.read_csv(dir1)
    adj_3 = adj_3.set_axis(list(adj_3), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v2.csv"
    segs_labels = pd.read_csv(dir2)
    labels_3 = segs_labels["3"]
    labels_3 = [x for x in labels_3 if str(x) != "nan"]

    return adj_3, labels_3


def load_left_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_left_with_class_v3.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemis_classes_v3.csv"
    hemi_labels = pd.read_csv(dir2)
    left_labels = hemi_labels["left"]
    left_labels = [x for x in left_labels if str(x) != "nan"]

    return left_adj, left_labels


def load_right_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_right_with_class_v3.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    dir2 = DATA_PATH / "labels_hemis_classes_v3.csv"
    hemi_labels = pd.read_csv(dir2)
    right_labels = hemi_labels["right"]
    right_labels = [x for x in right_labels if str(x) != "nan"]

    return right_adj, right_labels


def load_head_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_head_with_class_v3.csv"
    head_adj = pd.read_csv(dir1)
    head_adj = head_adj.set_axis(list(head_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v3.csv"
    segs_labels = pd.read_csv(dir2)
    head_labels = segs_labels["head"]
    head_labels = [x for x in head_labels if str(x) != "nan"]

    return head_adj, head_labels


def load_pygidium_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_pygidium_with_class_v3.csv"
    pyg_adj = pd.read_csv(dir1)
    pyg_adj = pyg_adj.set_axis(list(pyg_adj), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v3.csv"
    segs_labels = pd.read_csv(dir2)
    pyg_labels = segs_labels["pygidium"]
    pyg_labels = [x for x in pyg_labels if str(x) != "nan"]

    return pyg_adj, pyg_labels


def load_0_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_0_with_class_v3.csv"
    adj_0 = pd.read_csv(dir1)
    adj_0 = adj_0.set_axis(list(adj_0), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v3.csv"
    segs_labels = pd.read_csv(dir2)
    labels_0 = segs_labels["0"]
    labels_0 = [x for x in labels_0 if str(x) != "nan"]

    return adj_0, labels_0


def load_1_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_1_with_class_v3.csv"
    adj_1 = pd.read_csv(dir1)
    adj_1 = adj_1.set_axis(list(adj_1), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v3.csv"
    segs_labels = pd.read_csv(dir2)
    labels_1 = segs_labels["1"]
    labels_1 = [x for x in labels_1 if str(x) != "nan"]

    return adj_1, labels_1


def load_2_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_2_with_class_v3.csv"
    adj_2 = pd.read_csv(dir1)
    adj_2 = adj_2.set_axis(list(adj_2), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v3.csv"
    segs_labels = pd.read_csv(dir2)
    labels_2 = segs_labels["2"]
    labels_2 = [x for x in labels_2 if str(x) != "nan"]

    return adj_2, labels_2


def load_3_adj_labels_with_class_v3():
    dir1 = DATA_PATH / "adj_3_with_class_v3.csv"
    adj_3 = pd.read_csv(dir1)
    adj_3 = adj_3.set_axis(list(adj_3), axis="index")

    dir2 = DATA_PATH / "labels_segs_classes_v3.csv"
    segs_labels = pd.read_csv(dir2)
    labels_3 = segs_labels["3"]
    labels_3 = [x for x in labels_3 if str(x) != "nan"]

    return adj_3, labels_3


def load_graph_match_df():
    dir = DATA_PATH / "graph_match_df.csv"
    df = pd.read_csv(dir)
    return df


def load_graph_match_whole_df():
    dir = DATA_PATH / "graph_match_whole_df.csv"
    df = pd.read_csv(dir)
    return df


def load_graph_match_sub_df():
    dir = DATA_PATH / "graph_match_sub_df.csv"
    df = pd.read_csv(dir)
    return df


def load_best_left_adj_with_class():
    dir1 = DATA_PATH / "whole_best_left_adj.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    dir2 = DATA_PATH / "whole_best_left_labels.csv"
    labels_left = pd.read_csv(dir2)
    left_labels = labels_left["0"].tolist()

    return left_adj, left_labels


def load_best_right_adj_with_class():
    dir1 = DATA_PATH / "whole_best_right_adj.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    dir2 = DATA_PATH / "whole_best_right_labels.csv"
    labels_right = pd.read_csv(dir2)
    right_labels = labels_right["0"].tolist()

    return right_adj, right_labels


def load_best_left_adj_with_class_align():
    dir1 = DATA_PATH / "whole_best_left_adj_align.csv"
    left_adj = pd.read_csv(dir1)
    left_adj = left_adj.set_axis(list(left_adj), axis="index")

    dir2 = DATA_PATH / "whole_best_left_labels_align.csv"
    labels_left = pd.read_csv(dir2)
    left_labels = labels_left["0"].tolist()

    return left_adj, left_labels


def load_best_right_adj_with_class_align():
    dir1 = DATA_PATH / "whole_best_right_adj_align.csv"
    right_adj = pd.read_csv(dir1)
    right_adj = right_adj.set_axis(list(right_adj), axis="index")

    dir2 = DATA_PATH / "whole_best_right_labels_align.csv"
    labels_right = pd.read_csv(dir2)
    right_labels = labels_right["0"].tolist()

    return right_adj, right_labels


df = load_graph_match_whole_df()
print(df)
# annot = load_connectome_normal_lcc_annotations_v3()
# print(annot)
# annot_v2 = load_connectome_normal_lcc_annotations_v2()
# print(annot_v2)
# adj, labels = load_left_adj_labels_with_class_v2()
# print(len(adj))
# print(len(labels))
"""
print(len(load_left_adj_labels_with_class_v3()[0]))
print(len(load_left_adj_labels_with_class_v3()[1]))
print(len(load_right_adj_labels_with_class_v3()[0]))
print(len(load_right_adj_labels_with_class_v3()[1]))
print(len(load_head_adj_labels_with_class_v3()[0]))
print(len(load_head_adj_labels_with_class_v3()[1]))
print(len(load_pygidium_adj_labels_with_class_v3()[0]))
print(len(load_pygidium_adj_labels_with_class_v3()[1]))
print(len(load_1_adj_labels_with_class_v3()[0]))
print(len(load_1_adj_labels_with_class_v3()[1]))
print(len(load_2_adj_labels_with_class_v3()[0]))
print(len(load_2_adj_labels_with_class_v3()[1]))
print(len(load_3_adj_labels_with_class_v3()[0]))
print(len(load_3_adj_labels_with_class_v3()[1]))
print(load_connectome_normal_lcc_annotations_v3())
"""
# print(len(load_left_adj_labels_with_class_v2()[0]))
# print(load_right_adj_labels_with_class_v3()[0])