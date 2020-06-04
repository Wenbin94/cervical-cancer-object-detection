import kfbReader
from PIL import Image
import glob
import sys
from tqdm import tqdm
import numpy as np

def load_full_kfb(read):
    width = read.getWidth()
    height = read.getHeight()
    scale = 20
    step = 5000
    x = 0
    y = 0
    res = []
    res_row = []
    while y + step < height:
        res_row = []
        x = 0
        while x + step < width:
            tmp = read.ReadRoi(x, y, step, step, scale)
            res_row.append(tmp.copy())
            x += step
        res_row = np.asarray(res_row)
        res_row = np.concatenate(res_row, axis=1)
        # read last piece
        res_row = np.concatenate([res_row, read.ReadRoi(x, y, width - x, step, scale)], axis=1)
        res.append(res_row)
        y += step
    x = 0
    res = np.asarray(res)
    res = np.concatenate(res, axis=0)
    # read in last row
    res_row = []
    while x + step < width:
        tmp = read.ReadRoi(x, y, step, height - y, scale)
        res_row.append(tmp.copy())
        x += step
    res_row = np.asarray(res_row)
    res_row = np.concatenate(res_row, axis=1)
    # read last piece
    res_row = np.concatenate([res_row, read.ReadRoi(x, y, width - x, height - y, scale)], axis=1)
    res = np.concatenate([res, res_row], axis=0)
    im = Image.fromarray(res)
    return res


