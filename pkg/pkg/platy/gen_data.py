from operator import index
import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt
from graspologic.utils import largest_connected_component, is_fully_connected


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
    annotations.to_csv(path + "/all_annotations_new.csv", index_label="skids")
    return annotations


def gen_connectome_annotations():
    all_annotations = gen_all_annotations()
    skids_connec = pymaid.get_skids_by_annotation("connectome")
    skids_connec = [str(skid) for skid in skids_connec]
    skids_extra = []

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
    connec_lcc_annots.to_csv(path + "/connec_lcc_annotations_new.csv", index_label="skids")
    return connec_lcc_annots


def gen_normal_connectome_lcc_annotations():
    connec_lcc_annots = gen_connectome_lcc_annotations()
    print(connec_lcc_annots)
    inds = gen_weird_neurons_annots().index
    connec_lcc_normal_annots = connec_lcc_annots.loc[
        [i for i in connec_lcc_annots.index if i not in inds]
    ]
    connec_lcc_normal_annots.to_csv(
        path + "/connec_lcc_normal_annotations_new.csv", index_label="skids"
    )
    print(connec_lcc_normal_annots)
    return gen_normal_connectome_lcc_annotations
    


def gen_connectome_adj():
    skids_connec = pymaid.get_skids_by_annotation("connectome")
    adj_pandas = pymaid.adjacency_matrix(skids_connec)
    adj_pandas.to_csv(path + "/adj_connectome_new.csv", index=False)
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
    adj_lcc_pandas.to_csv(path + "/adj_connectome_lcc_new.csv", index=False)
    return adj_lcc_pandas


def gen_connectome_normal_lcc_adj():
    adj_lcc = gen_connectome_lcc_adj()
    adj_lcc_skids = adj_lcc.index
    adj_lcc_skids = [str(skid) for skid in adj_lcc_skids]
    inds = gen_weird_neurons_annots().index
    normal_skids = [int(i) for i in adj_lcc_skids if i not in inds]
    connec_normal_lcc_adj = adj_lcc.loc[normal_skids, normal_skids]
    connec_normal_lcc_adj.to_csv(path + "/adj_connectome_normal_lcc_new.csv", index=False)
    return connec_normal_lcc_adj


def gen_full_adj():
    all_skids = list(gen_all_annotations().index)
    full_adj = pymaid.adjacency_matrix(all_skids)
    full_adj.to_csv(path + "/full_adj_new.csv", index=False)
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
    weird_annots.to_csv(path + "/weird_annotations_new.csv", index_label="skids")
    return weird_annots

def gen_left_adj():
    skids_hemis = pd.read_csv(path + "/skids_hemi_classes.csv")
    skids_left = skids_hemis["l"]
    skids_left = [int(x) for x in skids_left if str(x) != "nan"]

    adj_left = pymaid.adjacency_matrix(skids_left)
    adj_left.to_csv(path + "/adj_left_new.csv", index=False)
    return adj_left

def gen_right_adj():
    skids_hemis = pd.read_csv(path + "/skids_hemi_classes.csv")
    skids_right = skids_hemis["r"]
    skids_right = [int(x) for x in skids_right if str(x) != "nan"]

    adj_right = pymaid.adjacency_matrix(skids_right)
    adj_right.to_csv(path + "/adj_right_new.csv", index=False)

    return adj_right

def gen_head_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_head = skids_segs["head"]
    skids_head = [int(x) for x in skids_head if str(x) != "nan"]

    adj_head = pymaid.adjacency_matrix(skids_head)
    adj_head.to_csv(path + "/adj_head_new.csv", index=False)

    return adj_head

def gen_pygidium_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_pyg = skids_segs["pygidium"]
    skids_pyg = [int(x) for x in skids_pyg if str(x) != "nan"]

    adj_pyg = pymaid.adjacency_matrix(skids_pyg)
    adj_pyg.to_csv(path + "/adj_pygidium_new.csv", index=False)

    return adj_pyg

def gen_0_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_0 = skids_segs["0"]
    skids_0 = [int(x) for x in skids_0 if str(x) != "nan"]

    adj_0 = pymaid.adjacency_matrix(skids_0)
    adj_0.to_csv(path + "/adj_0_new.csv", index=False)

    return adj_0

def gen_1_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_1 = skids_segs["1"]
    skids_1 = [int(x) for x in skids_1 if str(x) != "nan"]

    adj_1 = pymaid.adjacency_matrix(skids_1)
    adj_1.to_csv(path + "/adj_1_new.csv", index=False)

    return adj_1

def gen_2_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_2 = skids_segs["2"]
    skids_2 = [int(x) for x in skids_2 if str(x) != "nan"]

    adj_2 = pymaid.adjacency_matrix(skids_2)
    adj_2.to_csv(path + "/adj_2_new.csv", index=False)

    return adj_2

def gen_3_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_3 = skids_segs["3"]
    skids_3 = [int(x) for x in skids_3 if str(x) != "nan"]

    adj_3 = pymaid.adjacency_matrix(skids_3)
    adj_3.to_csv(path + "/adj_3_new.csv", index=False)

    return adj_3

def gen_left_normal_lcc_adj():
    skids_hemis = pd.read_csv(path + "/skids_hemi_classes.csv")
    skids_left = list(skids_hemis["l"])
    skids_left = [x for x in skids_left if str(x) != 'nan']
    skids_left = [int(i) for i in skids_left]

    left_connec_normal_adj = pymaid.adjacency_matrix(skids_left)
    left_connec_normal_adj_np = left_connec_normal_adj.to_numpy()
    left_connec_lcc_normal_adj, kept_inds = largest_connected_component(left_connec_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_left[ind]

    kept_skids = [int(i) for i in kept_skids]
    left_connec_lcc_normal_adj_pd = pd.DataFrame(left_connec_lcc_normal_adj, columns=kept_skids, index=kept_skids)
    
    left_connec_lcc_normal_adj_pd.to_csv(path + "/adj_left_normal_lcc_new.csv", index=False)
    return left_connec_lcc_normal_adj_pd

def gen_right_normal_lcc_adj():
    skids_hemis = pd.read_csv(path + "/skids_hemi_classes.csv")
    skids_right = list(skids_hemis["r"])
    skids_right = [x for x in skids_right if str(x) != 'nan']
    skids_right = [int(i) for i in skids_right]

    right_connec_normal_adj = pymaid.adjacency_matrix(skids_right)
    right_connec_normal_adj_np = right_connec_normal_adj.to_numpy()
    right_connec_lcc_normal_adj, kept_inds = largest_connected_component(right_connec_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_right[ind]

    kept_skids = [int(i) for i in kept_skids]
    right_connec_lcc_normal_adj_pd = pd.DataFrame(right_connec_lcc_normal_adj, columns=kept_skids, index=kept_skids)
    
    right_connec_lcc_normal_adj_pd.to_csv(path + "/adj_right_normal_lcc_new.csv", index=False)
    return right_connec_lcc_normal_adj_pd

def gen_head_normal_lcc_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_head = list(skids_segs["head"])
    skids_head = [x for x in skids_head if str(x) != 'nan']
    skids_head = [int(i) for i in skids_head]

    head_connec_normal_adj = pymaid.adjacency_matrix(skids_head)
    head_connec_normal_adj_np = head_connec_normal_adj.to_numpy()
    head_connec_lcc_normal_adj, kept_inds = largest_connected_component(head_connec_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_head[ind]

    kept_skids = [int(i) for i in kept_skids]
    head_connec_lcc_normal_adj_pd = pd.DataFrame(head_connec_lcc_normal_adj, columns=kept_skids, index=kept_skids)
    
    head_connec_lcc_normal_adj_pd.to_csv(path + "/adj_head_normal_lcc_new.csv", index=False)
    return head_connec_lcc_normal_adj_pd

def gen_pygidium_normal_lcc_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_pyg = list(skids_segs["pygidium"])
    skids_pyg = [x for x in skids_pyg if str(x) != 'nan']
    skids_pyg = [int(i) for i in skids_pyg]

    pyg_connec_normal_adj = pymaid.adjacency_matrix(skids_pyg)
    pyg_connec_normal_adj_np = pyg_connec_normal_adj.to_numpy()
    pyg_connec_lcc_normal_adj, kept_inds = largest_connected_component(pyg_connec_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_pyg[ind]

    kept_skids = [int(i) for i in kept_skids]
    pyg_connec_lcc_normal_adj_pd = pd.DataFrame(pyg_connec_lcc_normal_adj, columns=kept_skids, index=kept_skids)
    
    pyg_connec_lcc_normal_adj_pd.to_csv(path + "/adj_pygidium_normal_lcc_new.csv", index=False)
    return pyg_connec_lcc_normal_adj_pd

def gen_0_normal_lcc_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_0 = list(skids_segs["0"])
    skids_0 = [x for x in skids_0 if str(x) != 'nan']
    skids_0 = [int(i) for i in skids_0]

    connec_0_normal_adj = pymaid.adjacency_matrix(skids_0)
    connec_0_normal_adj_np = connec_0_normal_adj.to_numpy()
    connec_0_normal_adj_adj, kept_inds = largest_connected_component(connec_0_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_0[ind]

    kept_skids = [int(i) for i in kept_skids]
    connec_0_lcc_normal_adj_pd = pd.DataFrame(connec_0_normal_adj_adj, columns=kept_skids, index=kept_skids)
    
    connec_0_lcc_normal_adj_pd.to_csv(path + "/adj_0_normal_lcc_new.csv", index=False)
    return connec_0_lcc_normal_adj_pd

def gen_1_normal_lcc_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_1 = list(skids_segs["1"])
    skids_1 = [x for x in skids_1 if str(x) != 'nan']
    skids_1 = [int(i) for i in skids_1]

    connec_1_normal_adj = pymaid.adjacency_matrix(skids_1)
    connec_1_normal_adj_np = connec_1_normal_adj.to_numpy()
    connec_1_normal_adj_adj, kept_inds = largest_connected_component(connec_1_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_1[ind]

    kept_skids = [int(i) for i in kept_skids]
    connec_1_lcc_normal_adj_pd = pd.DataFrame(connec_1_normal_adj_adj, columns=kept_skids, index=kept_skids)
    
    connec_1_lcc_normal_adj_pd.to_csv(path + "/adj_1_normal_lcc_new.csv", index=False)
    return connec_1_lcc_normal_adj_pd

def gen_2_normal_lcc_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_2 = list(skids_segs["2"])
    skids_2 = [x for x in skids_2 if str(x) != 'nan']
    skids_2 = [int(i) for i in skids_2]

    connec_2_normal_adj = pymaid.adjacency_matrix(skids_2)
    connec_2_normal_adj_np = connec_2_normal_adj.to_numpy()
    connec_2_normal_adj_adj, kept_inds = largest_connected_component(connec_2_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_2[ind]

    kept_skids = [int(i) for i in kept_skids]
    connec_2_lcc_normal_adj_pd = pd.DataFrame(connec_2_normal_adj_adj, columns=kept_skids, index=kept_skids)
    
    connec_2_lcc_normal_adj_pd.to_csv(path + "/adj_2_normal_lcc_new.csv", index=False)
    return connec_2_lcc_normal_adj_pd

def gen_3_normal_lcc_adj():
    skids_segs = pd.read_csv(path + "/skids_segs_classes.csv")
    skids_3 = list(skids_segs["3"])
    skids_3 = [x for x in skids_3 if str(x) != 'nan']
    skids_3 = [int(i) for i in skids_3]

    connec_3_normal_adj = pymaid.adjacency_matrix(skids_3)
    connec_3_normal_adj_np = connec_3_normal_adj.to_numpy()
    connec_3_normal_adj_adj, kept_inds = largest_connected_component(connec_3_normal_adj_np, return_inds=True)

    kept_skids = np.ndarray(kept_inds.shape)
    for i, ind in enumerate(kept_inds):
        kept_skids[i] = skids_3[ind]

    kept_skids = [int(i) for i in kept_skids]
    connec_3_lcc_normal_adj_pd = pd.DataFrame(connec_3_normal_adj_adj, columns=kept_skids, index=kept_skids)
    
    connec_3_lcc_normal_adj_pd.to_csv(path + "/adj_3_normal_lcc_new.csv", index=False)
    return connec_3_lcc_normal_adj_pd

def gen_hemi_labels():
    #get skids for l, r, head, pygidium, 0, 1, 2, 3

    annotations = gen_connectome_normal_lcc_adj()

    left_adj = gen_left_normal_lcc_adj()
    left_adj_index = list(left_adj.index)

    right_adj = gen_right_normal_lcc_adj()
    right_adj_index = list(right_adj.index)

    all_hemi_index  = left_adj_index + right_adj_index

    annotations_hemis = annotations.loc[all_hemi_index]
    return len(all_hemi_index)
"""
    skids_hemis = {"l": [], "r": []}
    skids_hemis_total_list = []
    labels_hemis = {"l": [], "r": []}
    skids_segs = {"head": [], "pygidium": [], "0": [], "1": [], "2": [], "3": []}
    skids_segs_total_list = []
    labels_segs = {"head": [], "pygidium": [], "0": [], "1": [], "2": [], "3": []}
    poss_labels = ["s", "i", "m"]

#add skids and labels for hemis
    for key in skids_hemis:
        for i in range(len(annotations["skids"])):
            if(annotations["side"][i]==key and annotations["class"][i] in poss_labels):
                skids_hemis[key].append(annotations["skids"][i])
                skids_hemis_total_list.append(annotations["skids"][i])
                labels_hemis[key].append(annotations["class"][i])

    #add skids and labels for segments
    for key in skids_segs:
        for i in range(len(annotations["skids"])):
            if(annotations["segment"][i]==key and annotations["class"][i] in poss_labels):
                skids_segs[key].append(annotations["skids"][i])
                skids_segs_total_list.append(annotations["skids"][i])
                labels_segs[key].append(annotations["class"][i])

    skids_hemis_total_list = [str(x) for x in skids_hemis_total_list]

    #dictionaries to dfs
    skids_hemis_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in skids_hemis.items()]))
    labels_hemis_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in labels_hemis.items()]))
    skids_segs_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in skids_segs.items()]))
    labels_segs_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in labels_segs.items()]))

    #save dfs as csvs to get the adjs in gen_data
    skids_hemis_df.to_csv(folder / "skids_hemi_classes.csv")
    labels_hemis_df.to_csv(folder / "labels_hemi_classes.csv")
    skids_segs_df.to_csv(folder / "skids_segs_classes.csv")
    labels_segs_df.to_csv(folder / "labels_segs_classes.csv")
"""
print(gen_hemi_labels())