import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations

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
# type: ["Sensory neuron", "interneuron", "motorneuron"]
# segment: ["segment_0", "segment_1", "segment_2", "segment_3"]
# class: ["celltype1", "celltype2", ... "celltype180"]

# category can be "side" or "type" or "segment"
def get_labels_from_annotation(annot_list, category="side"):
    all_ids = pymaid.get_skids_by_annotation(annot_list)
    id_annot = []

    if category == "class":
        for annot in annot_list:
            ids = pymaid.get_skids_by_annotation(annot)
            for id in ids:
                if id in all_ids:
                    all_ids.remove(id)
                    label = annot.split("celltype")[1]
                    id_annot.append([id, label])

    # power set of annot_list reversed: first look at the intersections within annot_list then singular entries
    elif category == "side" or category == "type" or category == "segment":
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
                            elif category == "type":
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


side_list = ["left", "right", "center"]
type_list = ["Sensory neuron", "interneuron", "motorneuron"]
segment_list = ["segment_0", "segment_1", "segment_2", "segment_3"]

class_list = []

for i in range(1, 181):
    class_list.append("celltype{}".format(i))

side_labels = get_labels_from_annotation(side_list, category="side")
type_labels = get_labels_from_annotation(type_list, category="type")
segment_labels = get_labels_from_annotation(segment_list, category="segment")
class_labels = get_labels_from_annotation(class_list, category="class")
print(class_labels)

series_ids = [side_labels, type_labels, segment_labels, class_labels]
annotations = pd.concat(series_ids, axis=1, ignore_index=False, names="ID").fillna(
    "N/A"
)
print(annotations)
