
import kfbReader
from scripts.utils import load_full_kfb
from PIL import Image
import cv2 as cv

scale = 20
read = kfbReader.reader()
read.ReadInfo("./data/pos_0/T2019_156.kfb", scale, False)

#%%
size = 1024
res = load_full_kfb(read)
a = res
if a.shape[0] > a.shape[1]:
    width = size
    height = int(size * a.shape[0] / a.shape[1])
else:
    height = size
    width = int(size * a.shape[1] / a.shape[0])
a_small = cv.resize(a, (width, height))
#%%


#%%
size = 1024, 1024

#%%
