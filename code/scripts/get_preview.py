import kfbReader
import cv2 as cv
from PIL import Image
import glob
import sys
from tqdm import tqdm

scale = 20
for fp in tqdm(glob.glob(f"{sys.argv[1]}/*.kfb")):
    read = kfbReader.reader()
    read.ReadInfo(fp, scale, False)
    name = fp.replace(f"{sys.argv[1]}", "").replace(".kfb", "")
    preview = read.ReadPreview()
    im = Image.fromarray(preview[1].reshape(-1, preview[0], 3))
    im.save(f"{sys.argv[2]}/{name}_preview.jpg")

