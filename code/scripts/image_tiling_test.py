import kfbReader
import glob
import cv2 as cv
from tqdm import tqdm


scale = 20
size = 1000
overlap = 200
tile_df = {'file': [], 'x_start': [], 'y_start': [], 'w': [], 'h': []}

for fp in tqdm(glob.glob("./data/test/*.kfb")):
    read = kfbReader.reader()
    read.ReadInfo(fp, scale, False)
    width = read.getWidth()
    height = read.getHeight()

    # roi = H x W x C
    x, y = 0, 0
    counter = 0
    while y + size < height:
        x = 0
        while x + size < width:
            tile = read.ReadRoi(x, y, size,
                                size,
                                scale)
            cv.imwrite(f"./data/test_tiles/{fp.replace('./data/test/', '').replace('.kfb', '')}_{counter}.jpg", tile)
            tile_df['file'].append(f"{fp.replace('./data/test/', '').replace('.kfb', '')}_{counter}.jpg")
            tile_df['x_start'].append(x)
            tile_df['y_start'].append(y)
            tile_df['w'] = size
            tile_df['h'] = size
            counter += 1
            x += min((size - overlap), width - (x + size))
        y += min((size - overlap), height - (y + size))
