from os.path import dirname, join
from pathlib import Path
import pymaid
import logging
import pandas as pd
import numpy as np
from itertools import chain, combinations
from upsetplot import plot
from matplotlib import pyplot as plt

DATA_PATH = Path(__file__).parent.parent
DATA_PATH = DATA_PATH / "docs" / "outputs"


def _get_folder(path=None):
    if path == None:
        path = DATA_PATH
    folder = path
    return folder


def load_annotations():
    dir = DATA_PATH / "annotations.csv"
    annotations = pd.read_csv(dir, index_col=False)
    return annotations
