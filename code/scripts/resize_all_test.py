import pandas as pd
import os
import kfbReader
from scripts.utils import load_full_kfb
from PIL import Image
import glob
import cv2 as cv
from tqdm import tqdm
import os
import time

scale = 20
size = 1024

for fp in tqdm(glob.glob("./data/test/*")):
    f_name = fp.replace('./data/test/', '').replace('.kfb','.jpg')
    read = kfbReader.reader()
    read.ReadInfo(fp, scale, False)
    a = load_full_kfb(read)
    if a.shape[0] > a.shape[1]:
        width = size
        height = int(size * a.shape[0] / a.shape[1])
    else:
        height = size
        width = int(size * a.shape[1] / a.shape[0])
    a_small = cv.resize(a, (width, height))
    cv.imwrite(f"./data/test_1024/{f_name}", a_small)
print("finished.")
time.sleep(10)
os.system("sudo shutdown now")
