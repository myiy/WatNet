import glob
import os
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import random
import numpy as np

SEED = 42

CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')

PALETTE = [
    [0, 0, 0],          # background 黑色
    [0, 0, 255],        # building
    [255, 255, 0],      # road
    [255, 0, 0],        # water 红色
    [159, 129, 183],    # barren
    [0, 255, 0],        # forest
    [255, 195, 128],    # agricultural
]

WATER_CLASS_INDEX = CLASSES.index('water')  # 获取'water'类别的索引位置


# 定义一个函数，用于设置随机种子，以确保生成的随机数是可复现的
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# 定义一个函数，用于解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/LoveDA/Train/Rural/masks_png")
    parser.add_argument("--output-mask-dir", default="data/LoveDA/Train/Rural/masks_png_convert")
    return parser.parse_args()


# 定义一个函数，用于将标签图像转换为二值化的二维数组（0表示水，1表示其他）
def convert_label(mask):
    mask = (mask == WATER_CLASS_INDEX + 1).astype(np.uint8)  # Convert to binary mask (1 for water, 0 for others)
    return mask


# 定义一个函数，用于将二值化的标签图像转换为彩色图像
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    # mask_rgb[mask == 1] = PALETTE[WATER_CLASS_INDEX + 1]  # Set color for water
    mask_rgb[mask == 0] = PALETTE[0]    # 设置水的颜色
    mask_rgb[mask == 1] = PALETTE[3]    # 设置其他的颜色
    return mask_rgb


# 定义一个函数，用于处理标签图像的格式
def patch_format(inp):
    (mask_path, masks_output_dir) = inp
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    label = convert_label(mask)
    rgb_label = label2rgb(label.copy())
    rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    out_mask_path_rgb = os.path.join(masks_output_dir + '_rgb', "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path_rgb, rgb_label)

    out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path, label)


# 主程序入口
if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    masks_output_dir = args.output_mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        os.makedirs(masks_output_dir + '_rgb')

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images splitting spends: {} s'.format(split_time))
