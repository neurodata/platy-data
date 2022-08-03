import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt

rm = pymaid.CatmaidInstance(
    server="https://catmaid.jekelylab.ex.ac.uk/#",
    project_id=11,
    api_token=None,
    http_user=None,
    http_password=None,
)
logging.getLogger("pymaid").setLevel(logging.WARNING)
pymaid.clear_cache()

# side: ["left", "right", "center"]
# class: ["Sensory neuron", "interneuron", "motorneuron"]
# segment: ["segment_0", "segment_1", "segment_2", "segment_3"]
# type: ["celltype1", "celltype2", ... "celltype180"]
# group: ["cellgroup1", "cellgroup2", ... "cellgroup18"]
# category can be "side" or "type" or "segment" or "class" or "group"
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
                                label += annot[-1]
                            else:
                                raise ValueError("category is invalid")
                        id_annot.append([id, label])

    else:
        raise ValueError("category is invalid")

    id_annot = np.array(id_annot)

    ids = id_annot[:, 0]
    annots = id_annot[:, 1]
    return pd.Series(index=ids, data=annots, name=category)


def has_category(annotations, category="side"):
    has_list = []
    series_name = "has_{}".format(category)
    for val in annotations[category]:
        if val != "N/A":
            has_list.append(True)
        else:
            has_list.append(False)
    return pd.Series(index=annotations.index, data=has_list, name=series_name)


side_list = ["left", "right", "center"]
class_list = ["Sensory neuron", "interneuron", "motorneuron"]
segment_list = ["segment_0", "segment_1", "segment_2", "segment_3"]

type_list = []
for i in range(1, 181):
    type_list.append("celltype{}".format(i))

group_list = []
for j in range(1, 18):
    group_list.append("cellgroup{}".format(j))


side_labels = get_labels_from_annotation(side_list, category="side")
class_labels = get_labels_from_annotation(class_list, category="class")
segment_labels = get_labels_from_annotation(segment_list, category="segment")
type_labels = get_labels_from_annotation(type_list, category="type")
group_labels = get_labels_from_annotation(group_list, category="group")

series_ids = [side_labels, class_labels, segment_labels, type_labels, group_labels]
annotations = pd.concat(series_ids, axis=1, ignore_index=False, names="ID").fillna(
    "N/A"
)

has_side = has_category(annotations, category="side")
has_class = has_category(annotations, category="class")
has_segment = has_category(annotations, category="segment")
has_type = has_category(annotations, category="type")
has_group = has_category(annotations, category="group")

bool_ids = [has_side, has_class, has_segment, has_type, has_group]
annotations_bool = pd.concat(bool_ids, axis=1, ignore_index=False, names="ID").fillna(
    "N/A"
)

bool_counts = annotations_bool.groupby(
    [has_side, has_type, has_segment, has_class, has_group]
).size()

plot(bool_counts)
plt.show()
