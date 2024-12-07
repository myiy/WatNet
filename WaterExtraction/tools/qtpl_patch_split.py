import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
import random
import glob


# 将遥感图像数据进行切割和数据增强，生成用于训练或验证的小图像块和对应的标签


SEED = 42


# 设置随机种子，以确保在训练模型等涉及到随机性的任务中，每次运行时生成的随机数和结果都是可复现的

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 将标签图像映射为RGB格式的图像
Water = np.array([255, 255, 255])  # 白色, label 0
Clutter = np.array([0, 0, 0])  # 黑色, label 1
num_classes = 2


# 目的是从命令行获取输入的参数值
def parse_args():
    parser = argparse.ArgumentParser()
    # 输入图像所在的目录路径
    parser.add_argument("--input-img-dir", default="data/QTPL/Train/Images")
    # 输入标签所在的目录路径
    parser.add_argument("--input-mask-dir", default="data/QTPL/Train/Labels")
    # 输出图像所在的目录路径
    parser.add_argument("--output-img-dir", default="data/QTPL/Train/images_png")
    # 输出标签图像（masks）所在的目录路径
    parser.add_argument("--output-mask-dir", default="data/QTPL/Train/masks_png")
    # 指定模式，类型为字符串，默认值为 'train'
    parser.add_argument("--mode", type=str, default='train')
    # 垂直方向上的分割大小
    parser.add_argument("--split-size-h", type=int, default=256)
    # 水平方向上的分割大小
    parser.add_argument("--split-size-w", type=int, default=256)
    # 垂直方向上的步幅
    parser.add_argument("--stride-h", type=int, default=256)
    # 水平方向上的步幅
    parser.add_argument("--stride-w", type=int, default=256)
    return parser.parse_args()


# 将标签图像（mask）转换为 RGB 彩色图像
def label2rgb(mask):
    # 获取标签图像的高度和宽度
    h, w = mask.shape[0], mask.shape[1]
    # 创建一个全零的三通道数组，表示输出的 RGB 彩色图像
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    # 在通道维度上添加一个维度，使得 mask_convert 的形状为 (1, h, w)
    mask_convert = mask[np.newaxis, :, :]
    # 将标签为 0 的区域赋值为 Water 的颜色
    mask_rgb[np.all(mask_convert == 0, axis=0)] = Water
    # 将标签为 1 的区域赋值为 Clutter 的颜色
    mask_rgb[np.all(mask_convert == 1, axis=0)] = Clutter

    # 返回 RGB 彩色图像
    return mask_rgb


# 将输入的 RGB 彩色图像转换为标签图像，其中每个像素的标签值由颜色决定。
def rgb2label(label):
    # 创建一个与输入 RGB 图像形状相同的全零数组，用于表示标签图像
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    # 将 RGB 图像中颜色为 Water 的像素赋值为标签值 0
    label_seg[np.all(label == Water, axis=-1)] = 0
    # 将 RGB 图像中颜色为 Clutter 的像素赋值为标签值
    label_seg[np.all(label == Clutter, axis=-1)] = 1

    # 返回标签图像
    return label_seg


# 用于对输入的图像和标签进行处理
# 将它们添加到相应的列表中，并最终返回处理后的图像和标签列表。
def image_augment(image, mask, mode='train'):
    # 初始化空列表，用于存储处理后的图像和标签
    image_list = []
    mask_list = []

    # 获取输入图像和标签的宽度和高度
    image_width, image_height = image.shape[1], image.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]

    # 断言确保输入图像和标签的尺寸一致
    assert image_height == mask_height and image_width == mask_width

    # print(f"Before augmentation - Image shape: {image.shape}, Mask shape: {mask.shape}")

    # 根据 mode 的值进行不同的处理
    if mode == 'train':
        # 在训练模式下，对原始图像和标签进行处理
        image_list_train = [image]
        mask_list_train = [mask]
        for i in range(len(image_list_train)):
            # 将标签图像转换为标签值
            mask_tmp = rgb2label(mask_list_train[i])
            # 将处理后的图像和标签添加到列表中
            image_list.append(image_list_train[i])
            mask_list.append(mask_tmp)
    else:
        # 在非训练模式下，对原始图像和标签进行处理
        mask = rgb2label(mask.copy())
        # 将处理后的图像和标签添加到列表中
        image_list.append(image)
        mask_list.append(mask)

    # print(f"After augmentation - Image list length: {len(image_list)}, Mask list length: {len(mask_list)}")

    # 返回处理后的图像和标签列表
    return image_list, mask_list


# 对图像和标签进行填充，确保输入的图像和标签满足特定的尺寸要求，以便进行后续的处理。
def padifneeded(image, mask, patch_size, stride):
    # 获取输入图像的高度和宽度
    oh, ow = image.shape[0], image.shape[1]

    # 初始化填充高度和宽度
    padh, padw = 0, 0

    # 循环直到高度满足 patch_size[0] 和 stride[0] 的条件，逐步增加填充高度。
    while (oh + padh -patch_size[0]) % stride[0] != 0:
        padh = padh + 1

    # 循环直到宽度满足 patch_size[1] 和 stride[1] 的条件，逐步增加填充宽度。
    while (ow + padw -patch_size[1]) % stride[1] != 0:
        padw = padw + 1

    # 计算最终的高度和宽度
    h, w = oh + padh, ow + padw

    # 使用 albumentations 库的 PadIfNeeded 函数进行填充
    pad = albu.PadIfNeeded(min_height=h, min_width=w)(image=image, mask=mask)
    img_pad, mask_pad = pad['image'], pad['mask']

    # 返回填充后的图像和标签
    # print(img_pad.shape)
    return img_pad, mask_pad


def patch_format(inp):
    # 解包输入元组
    (img_path, mask_path, imgs_output_dir, masks_output_dir, mode, split_size, stride) = inp
    if mode == 'val':
        gt_path = masks_output_dir + "_gt"

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    id = os.path.splitext(os.path.basename(img_path))[0]
    assert img.shape == mask.shape

    img, mask = padifneeded(img.copy(), mask.copy(), split_size, stride)

    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), mode=mode)
    assert len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
        for y in range(0, img.shape[0], stride[0]):
            for x in range(0, img.shape[1], stride[1]):
                img_tile_cut = img[y:y + split_size[0], x:x + split_size[1]]
                mask_tile_cut = mask[y:y + split_size[0], x:x + split_size[1]]
                img_tile, mask_tile = img_tile_cut, mask_tile_cut

                if img_tile.shape[0] == split_size[0] and img_tile.shape[1] == split_size[1] \
                        and mask_tile.shape[0] == split_size[0] and mask_tile.shape[1] == split_size[1]:
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_tile, bins=bins)
                    # cf = class_pixel_counts / (mask_tile.shape[0] * mask_tile.shape[1])
                    # if cf[0] > 0.00:
                    if mode == 'train':
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)
                    else:
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                        out_mask_path_gt = os.path.join(gt_path, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_mask_path_gt, label2rgb(mask_tile))

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()

    input_img_dir = args.input_img_dir
    input_mask_dir = args.input_mask_dir

    img_paths = glob.glob(os.path.join(input_img_dir, "*.png"))
    mask_paths = glob.glob(os.path.join(input_mask_dir, "*.png"))

    img_paths.sort()
    mask_paths.sort()

    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir

    mode = args.mode

    split_size_h = args.split_size_h
    split_size_w = args.split_size_w
    split_size = (split_size_h, split_size_w)
    stride_h = args.stride_h
    stride_w = args.stride_w
    stride = (stride_h, stride_w)

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, mode, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))





# def patch_format(inp):
#     # 解包输入元组
#     (input_dir, imgs_output_dir, masks_output_dir, mode, split_size, stride) = inp
#     # 获取所有图像和标签的文件路径
#     img_paths = glob.glob(os.path.join(input_dir, 'Images', "*.png"))
#     mask_paths = glob.glob(os.path.join(input_dir, 'Labels', "*.png"))
#     # 遍历图像和标签路径
#     for img_path, mask_path in zip(img_paths, mask_paths):
#         # 读取图像和标签
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
#         # 将图像和标签转换为RGB格式
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#         # 获取图像的文件名（不包含扩展名），作为标识符
#         id = os.path.splitext(os.path.basename(img_path))[0]
#         # 断言确保图像和标签的形状一致
#         assert img.shape == mask.shape
#         # 对图像和标签进行填充处理
#         img, mask = padifneeded(img.copy(), mask.copy(), split_size, stride)
#
#         # 调用 image_augment 函数，对图像和标签进行处理
#         image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), mode=mode)
#         # 使用断言确保处理后的图像和标签列表长度一致
#         assert len(image_list) == len(mask_list)
#
#         # 遍历处理后的图像和标签列表
#         for m in range(len(image_list)):
#             # 初始化索引 k 为 0
#             k = 0
#             # 获取当前索引 m 对应的图像和标签
#             img = image_list[m]
#             mask = mask_list[m]
#             # 使用断言确保当前图像和标签的高度和宽度一致
#             assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
#             # 循环遍历图像，按照给定的步幅和分割大小，提取图像块和相应的标签块
#             for y in range(0, img.shape[0], stride[0]):
#                 for x in range(0, img.shape[1], stride[1]):
#                     # 提取当前位置的图像块和相应的标签块
#                     img_tile_cut = img[y:y + split_size[0], x:x + split_size[1]]
#                     mask_tile_cut = mask[y:y + split_size[0], x:x + split_size[1]]
#                     # 将提取的图像块和标签块赋值给新的变量
#                     img_tile, mask_tile = img_tile_cut, mask_tile_cut
#
#                     if img_tile.shape[0] == split_size[0] and img_tile.shape[1] == split_size[1] \
#                             and mask_tile.shape[0] == split_size[0] and mask_tile.shape[1] == split_size[1]:
#                         bins = np.array(range(num_classes + 1))
#                         class_pixel_counts, _ = np.histogram(mask_tile, bins=bins)
#                         # 当前块是否包含某个类别的像素，进而决定是否保存该块。
#                         # cf = class_pixel_counts / (mask_tile.shape[0] * mask_tile.shape[1])
#                         # if cf[0] > 0.00:
#                         if mode == 'train':
#                             # 将 RGB 格式的图像块转换为 BGR 格式
#                             img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
#                             # 构建输出图像文件的路径，其中包含图像的ID、m和k
#                             out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(id, m, k))
#                             # 保存图像块到指定路径
#                             cv2.imwrite(out_img_path, img_tile)
#
#                             # 构建输出标签块文件的路径，其中包含图像的ID、m和k
#                             out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(id, m, k))
#                             # 保存标签块到指定路径。
#                             cv2.imwrite(out_mask_path, mask_tile)
#                         # 如果处理模式不是训练模式
#                         else:
#                             img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
#                             out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(id, m, k))
#                             cv2.imwrite(out_img_path, img_tile)
#
#                             out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(id, m, k))
#                             cv2.imwrite(out_mask_path, mask_tile)
#
#                     k += 1


# if __name__ == "__main__":
#     # 设置随机种子以保证可复现性
#     seed_everything(SEED)
#     args = parse_args()
#
#     input_dir = args.input_dir
#
#     imgs_output_dir = args.output_img_dir
#     masks_output_dir = args.output_mask_dir
#
#     mode = args.mode
#
#     split_size_h = args.split_size_h
#     split_size_w = args.split_size_w
#     split_size = (split_size_h, split_size_w)
#     stride_h = args.stride_h
#     stride_w = args.stride_w
#     stride = (stride_h, stride_w)
#     # seqs = os.listdir(input_dir)
#
#     # 创建输出目录
#     if not os.path.exists(imgs_output_dir):
#         os.makedirs(imgs_output_dir)
#     if not os.path.exists(masks_output_dir):
#         os.makedirs(masks_output_dir)
#
#     # 构建输入列表
#     inp = [(input_dir, imgs_output_dir, masks_output_dir, mode, split_size, stride)
#            ]
#
#     # 并行处理任务
#     t0 = time.time()
#     mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
#     t1 = time.time()
#     split_time = t1 - t0
#     print('images spliting spends: {} s'.format(split_time))
#
