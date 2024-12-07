import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random

# 用于图像的可视化，白色可能表示水，黑色表示背景
CLASSES = ('Water', 'Clutter')
PALETTE = [[255, 255, 255],  [0, 0, 0]]

# 定义原始图像、输入图像、测试图像的大小为 (512, 512)
ORIGIN_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (256, 256)
TEST_IMG_SIZE = (256, 256)

# 获取训练数据增强的转换


def get_training_transform():
    train_transform = [
        # 添加了一个水平翻转的数据增强，概率为 0.5
        albu.HorizontalFlip(p=0.5),
        # 添加了一个垂直翻转的数据增强，概率为 0.5
        albu.VerticalFlip(p=0.5),
        # 添加了一个随机旋转 90 度的数据增强，概率为 0.5
        albu.RandomRotate90(p=0.5),
        # albu.RandomBrightnessContrast(p=0.5),
        # 图像归一化
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


# 确保输入数据在与训练时相同的尺度和范围上，以便模型在验证时能够得到一致的输入
def get_validation_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


# 预处理输入数据，以便它们能够适应模型的输入要求
def get_test_transform():
    test_transform = [
        albu.Normalize()
    ]
    return albu.Compose(test_transform)


class WhdldDataset(Dataset):
    # 初始化方法，设置数据集的参数
    def __init__(self, data_root='data/WHDLD/Train', mode='train', img_dir='images_png', mask_dir='masks_png',
                 img_suffix='.png', mask_suffix='.png', transform=None, mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root      # 数据集的根目录
        self.img_dir = img_dir          # 图像存储的目录
        self.mask_dir = mask_dir        # 掩码存储的目录
        self.img_suffix = img_suffix    # 图像文件的后缀
        self.mask_suffix = mask_suffix  # 掩码文件的后缀
        self.transform = transform      # 数据增强的方法
        self.mode = mode                # 模式，可以是'train'、'val'或'test'
        self.mosaic_ratio = mosaic_ratio    # 使用mosaic增强的比例
        self.img_size = img_size        # 图像的大小
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)    # 获取图像标识符列表

    def __getitem__(self, index):
        p_ratio = random.random()
        # 如果随机概率大于设定的 Mosaic 比例或者模式为 'val' 或 'test'
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            # 加载单个图像和相应的掩码
            img, mask = self.load_img_and_mask(index)
            # 如果定义了数据转换操作
            if self.transform:
                # 应用数据转换操作到图像和掩码
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
        # 如果随机概率小于等于设定的 Mosaic 比例，且模式不是 'val' 或 'test'
        else:
            # 加载 Mosaic 图像和相应的掩码
            img, mask = self.load_mosaic_img_and_mask(index)
            # 如果定义了数据转换操作
            if self.transform:
                # 应用数据转换操作到 Mosaic 图像和掩码
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            # else:
            #     # 如果没有定义数据转换操作，则将图像和掩码转换为 NumPy 数组
            #     img, mask = np.array(img), np.array(mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id}
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        # 断言图像文件和掩码文件的数量相同
        assert len(img_filename_list) == len(mask_filename_list)
        # 从掩码文件名列表中提取图像标识符，通常是文件名去除扩展名部分
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    # def load_img_and_mask(self, index):
    #     img_id = self.img_ids[index]
    #     # 构建图像和掩码的文件路径
    #     img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
    #     mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
    #     # 使用PIL库打开图像，并将其转换为RGB格式
    #     img = Image.open(img_name).convert('RGB')
    #     # 使用PIL库打开掩码，并将其转换为灰度格式
    #     mask = Image.open(mask_name).convert('L')
    #     # 返回加载的图像和掩码
    #     return img, mask

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.float32)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        # 随机选择4个索引，包括当前索引
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        # 分别加载4个图像和相应掩码
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        # 将加载的图像和掩码转换为NumPy数组
        # img_a, mask_a = np.array(img_a), np.array(mask_a)
        # img_b, mask_b = np.array(img_b), np.array(mask_b)
        # img_c, mask_c = np.array(img_c), np.array(mask_c)
        # img_d, mask_d = np.array(img_d), np.array(mask_d)

        # 获取图像尺寸
        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)
        # print(img.shape)

        return img, mask