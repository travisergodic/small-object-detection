import shutil
from pathlib import Path

import cv2
import imagesize
from tqdm import tqdm
import pandas as pd


EXTENSIONS = ('.jpg', 'png', '.tif')


def convert_to_yolov5(image_dir, anno_dir, dst, format_string='{:,.5f}'):
    Path(f'{dst}/labels').mkdir(exist_ok=True, parents=True)
    Path(f'{dst}/images').mkdir(exist_ok=True, parents=True)
    
    for img_path in tqdm(Path(image_dir).glob('*')):
        if not str(img_path).endswith(EXTENSIONS):  
            continue

        anno_path = str(Path(anno_dir) / (img_path.name.rsplit('.', 1)[0] + '.txt')) 

        if not Path(anno_path).is_file():
            continue

        W, H = imagesize.get(str(img_path))

        # read annotation
        anno_df = pd.read_csv(anno_path, names=['class', 'x', 'y', 'w', 'h'])
        
        # filter area = 0 bbox
        anno_df = anno_df.loc[(anno_df['w'] > 0) & (anno_df['h'] > 0), :].copy()
        
        anno_df['x_c'] = ((anno_df['x'] + anno_df['w']/2) / W)
        anno_df['y_c'] = (anno_df['y'] + anno_df['h']/2) / H
        anno_df['w'] = anno_df['w'] / W
        anno_df['h'] = anno_df['h'] / H

        # format string
        anno_df['x_c'] = anno_df['x_c'].map(format_string.format)
        anno_df['y_c'] = anno_df['y_c'].map(format_string.format)
        anno_df['w'] = anno_df['w'].map(format_string.format)
        anno_df['h'] = anno_df['h'].map(format_string.format)
        
        # save yolo format annotation
        shutil.copy(str(img_path), f'{dst}/images/{img_path.name}')
        anno_df[['class', 'x_c', 'y_c', 'w', 'h']].to_csv(
            f'{dst}/labels/{Path(anno_path).name}', index=False, sep=' ', header=False
        )


def draw_bbox(img_path, anno_path, class_to_color, thickness=2):
    img = cv2.imread(img_path)
    df = pd.read_csv(
        anno_path, sep=',', names=['class', 'x', 'y', 'w', 'h']
    ).astype(int)
    
    for _, row in df.iterrows():
        cls, x, y, w, h = row['class'], row['x'], row['y'], row['w'], row['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), class_to_color[cls], thickness)
    return img