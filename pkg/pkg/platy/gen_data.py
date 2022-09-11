from operator import index
import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt
from graspologic.utils import largest_connected_component


rm = pymaid.CatmaidInstance(
    server="https://catmaid.jekelylab.ex.ac.uk/#",
    project_id=11,
    api_token=None,
    http_user=None,
    http_password=None,
)
logging.getLogger("pymaid").setLevel(logging.WARNING)
pymaid.clear_cache()

path = "/Users/kareefullah/Desktop/neurodata/neurodata/platy-data/docs/outputs"

# side: ["left", "right", "center"]
# class: ["Sensory neuron", "interneuron", "motorneuron"]
# segment: ["segment_0", "segment_1", "segment_2", "segment_3", "head", "pygidium"]
# type: ["celltype1", "celltype2", ... "celltype180"]
# group: ["cellgroup1", "cellgroup2", ... "cellgroup18"]
# category can be "side" or "class" or "segment" or "type" or "group"

# store metadata - connectivity as weighted edgelist - node ids are skids, node data as csv (df_to_csv)
# print out # nodes, # edges, density, modularity, avg degree
# replicate figure 1
# graph matching (predict neuron pairs then look at morphologies), symmetry (cell type grouping/segments), sbm testing, clustering
# check for head/trunk/pygidium label (add to segment)
# add all figs to jupyter notebook

# save data (weighted edgelist and all the figs)
# clean up upsetplot stuff
# clean up fig 1
# density of left vs density of right/other density tests (tell Ben that I am at this point and I should talk to Jeremy)
# SBM test notebook


def get_labels_from_annotation(annot_list, category="side"):
    all_ids = pymaid.get_skids_by_annotation(annot_list)
    id_annot = []
    if category == "type" or category == "group":
        for annot in annot_list:
            ids = pymaid.get_skids_by_annotation(annot)
            for id in ids:
                if id in all_ids:
                    all_ids.remove(id)
                    if category == "type":
                        label = annot.split("celltype")[1]
                    else:
                        label = annot.split("cellgroup")[1]
                    id_annot.append([id, label])

    # power set of annot_list reversed: first look at the intersections within annot_list then singular entries
    elif category == "side" or category == "class" or category == "segment":
        annot_power = list(
            chain.from_iterable(
                combinations(annot_list, r) for r in range(len(annot_list) + 1)
            )
        )
        annot_power.reverse()

        for annots in annot_power:

            # skip the empty set in the power set
            if len(annots) != 0:
                ids = pymaid.get_skids_by_annotation(annots, intersect=True)

                # assign label to every id
                for id in ids:
                    if id in all_ids:
                        # make sure cannot find id again if it is already added to id_list
                        all_ids.remove(id)
                        label = ""
                        for annot in annots:
                            if category == "side":
                                label += annot[0]
                            elif category == "class":
                                label += annot[0].lower()
                            elif category == "segment":
                                if annot[0] == "s":
                                    label += annot[-1]
                                else:
                                    label += annot
                            else:
                                raise ValueError("category is invalid")
                        id_annot.append([id, label])

    else:
        raise ValueError("category is invalid")

    id_annot = np.array(id_annot)

    ids = id_annot[:, 0]
    annots = id_annot[:, 1]
    return pd.Series(index=ids, data=annots, name=category)


def gen_all_annotations():
    side_list = ["left", "right", "center"]
    class_list = ["Sensory neuron", "interneuron", "motorneuron"]
    segment_list = [
        "segment_0",
        "segment_1",
        "segment_2",
        "segment_3",
        "head",
        "pygidium",
    ]

    type_list = []
    for i in range(1, 181):
        type_list.append("celltype{}".format(i))

    group_list = []
    for j in range(1, 19):
        group_list.append("cellgroup{}".format(j))

    side_labels = get_labels_from_annotation(side_list, category="side")
    class_labels = get_labels_from_annotation(class_list, category="class")
    segment_labels = get_labels_from_annotation(segment_list, category="segment")
    type_labels = get_labels_from_annotation(type_list, category="type")
    group_labels = get_labels_from_annotation(group_list, category="group")

    series_ids = [
        side_labels,
        class_labels,
        segment_labels,
        type_labels,
        group_labels,
    ]
    annotations = pd.concat(series_ids, axis=1, ignore_index=False, names="ID").fillna(
        "N/A"
    )
    annotations.to_csv(path + "/all_annotations.csv", index_label="skids")
    return annotations


def gen_connectome_annotations():
    all_annotations = gen_all_annotations()
    skids_connec = pymaid.get_skids_by_annotation("connectome")
    skids_connec = [str(skid) for skid in skids_connec]
    skids_extra = []
    count = 0
    for skid in skids_connec:
        if skid not in all_annotations.index:
            skids_extra.append(skid)

    for skid in skids_extra:
        all_annotations.loc[skid] = [
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        ]
    connec_annots = all_annotations.loc[skids_connec]
    # connec_annots.to_csv(path + "/connec_annotations.csv", index_label="skids")
    return connec_annots


def gen_connectome_lcc_annotations():
    connec_annots = gen_connectome_annotations()
    inds = gen_connectome_lcc_adj().index
    inds = [str(skid) for skid in inds]
    connec_lcc_annots = connec_annots.loc[inds]
    connec_lcc_annots.to_csv(path + "/connec_lcc_annotations.csv", index_label="skids")
    return connec_lcc_annots


def gen_connectome_adj():
    skids_connec = pymaid.get_skids_by_annotation("connectome")
    adj_pandas = pymaid.adjacency_matrix(skids_connec)
    adj_pandas.to_csv(path + "/adj_connectome.csv", index=False)
    return adj_pandas


def gen_connectome_lcc_adj():
    adj_connec = gen_connectome_adj()
    all_skids = adj_connec.index
    adj_connec_np = adj_connec.to_numpy()
    adj_lcc, kept_inds = largest_connected_component(adj_connec_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = all_skids[ind]
    kept_skids = [int(i) for i in kept_skids]
    adj_lcc_pandas = pd.DataFrame(adj_lcc, columns=kept_skids, index=kept_skids)
    adj_lcc_pandas.to_csv(path + "/adj_connectome_lcc.csv", index=False)
    return adj_lcc_pandas


def gen_full_adj():
    all_skids = list(gen_all_annotations().index)
    full_adj = pymaid.adjacency_matrix(all_skids)
    full_adj.to_csv(path + "/full_adj.csv", index=False)
    return full_adj


def gen_weird_neurons_annots():
    connec_lcc_annots = gen_connectome_lcc_annotations()
    # check lr, lc, rc, im, sm
    weird_annots = connec_lcc_annots.loc[
        (connec_lcc_annots["side"] == "lr")
        | (connec_lcc_annots["side"] == "lc")
        | (connec_lcc_annots["side"] == "rc")
        | (connec_lcc_annots["class"] == "im")
        | (connec_lcc_annots["class"] == "sm")
    ]
    weird_annots.to_csv(path + "/weird_annotations.csv", index_label="skids")
    return weird_annots
