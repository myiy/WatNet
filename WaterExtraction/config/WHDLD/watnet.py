from torch.utils.data import DataLoader
from geoseg.datasets.whdld_dataset import *
from geoseg.losses import *
from geoseg.models.ABCNet import ABCNet
from geoseg.models.BANet import BANet
from geoseg.models.UNet import Unet
from geoseg.models.PSPNet import PSPNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# 训练中的一些超参数和配置
max_epoch = 100                  # 训练的最大轮次
ignore_index = len(CLASSES)     # 掩码图像中的背景
train_batch_size = 16           # 训练时每个批次的样本数量
val_batch_size = 16             # 验证时每个批次的样本数量
lr = 1e-4                       # 梯度下降中用于更新模型参数的步长
weight_decay = 0.01             # 权重衰减是正则化项，用于防止模型过拟合
backbone_lr = 1e-4              # 模型的骨干网络（backbone）的学习率
backbone_weight_decay = 0.01    # 模型的骨干网络的权重衰减
accumulate_n = 1                # 梯度累积，不会占用额外的内存
num_classes = len(CLASSES)      # 类别的数量
classes = CLASSES               # 类别列表


weights_name = "Wat-r50-492-768crop-e100"
weights_path = "model_weights/whdld/{}".format(weights_name)
test_weights_name = "last"
log_name = 'whdld/{}'.format(weights_name)

monitor = 'val_mIoU'            # 模型性能的评估指标，使用平均IoU
monitor_mode = 'max'            # 模型性能监控模式,max表示监控指标的值越大越好
save_top_k = 1                  # 保存最好的模型权重的数量
save_last = True                # 在训练结束后保存最后一轮的模型权重
check_val_every_n_epoch = 1     # 每个 epoch 结束后都在验证集上进行性能评估
gpus = 'auto'                   # 使用所有可能的 GPU 设备
strategy = None                 # 没有指定训练策略
pretrained_ckpt_path = None     # 预训练模型权重的路径
resume_ckpt_path = None         # 恢复训练时，指定之前保存的模型权重的路径

#  define the network
net = WatNet(num_classes=num_classes)
# define the loss
loss = EdgeLoss(ignore_index=ignore_index)

use_aux_loss = False


# define the dataloader
train_dataset = WhdldDataset(data_root='data/WHDLD/Train', mode='train', mosaic_ratio=0.25, transform=get_training_transform())

val_dataset = WhdldDataset(data_root='data/WHDLD/Val', mode='val', transform=get_validation_transform())
test_dataset = WhdldDataset(data_root='data/WHDLD/Val', mode='val', transform=get_test_transform())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=16,
                          pin_memory=True,
                          shuffle=False,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=16,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)



