"""
This script splits the data into 2 by different resolutions
"""
import pandas as pd
import os
from PIL import Image
import glob
import cv2 as cv
from tqdm import trange
import json
import numpy as np

WIDTH_THRESHOLD = 38000

# train
all_pos_sizes = pd.read_csv("./analyses/all_pos_sizes.csv")
all_pos_sizes_small = all_pos_sizes[all_pos_sizes['width'] < WIDTH_THRESHOLD]
all_pos_sizes_large = all_pos_sizes[all_pos_sizes['width'] >= WIDTH_THRESHOLD]
all_pos_sizes_small.to_csv("./analyses/all_pos_sizes_small.csv", index=False)
all_pos_sizes_large.to_csv("./analyses/all_pos_sizes_large.csv", index=False)

# test
all_test_sizes = pd.read_csv("./analyses/all_test_sizes.csv")
all_test_sizes_small = all_test_sizes[all_test_sizes['width'] < WIDTH_THRESHOLD]
all_test_sizes_large = all_test_sizes[all_test_sizes['width'] >= WIDTH_THRESHOLD]
all_test_sizes_small.to_csv("./analyses/all_test_sizes_small.csv", index=False)
all_test_sizes_large.to_csv("./analyses/all_test_sizes_large.csv", index=False)
