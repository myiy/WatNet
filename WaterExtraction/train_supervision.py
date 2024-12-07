import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random


# 设置随机种子，以确保深度学习模型的训练过程在每次运行时都是可重复的
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 创建一个命令行参数解析器，用于从命令行中获取一个配置文件的路径
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    # 初始化方法，接受一个名为 config 的参数，通常包含模型训练所需的各种配置信息
    def __init__(self, config):

        super().__init__()
        self.config = config
        # 从配置对象中获取网络模型（net）并保存在类的属性中
        self.net = config.net
        # 将传入的配置对象保存在类的属性中，以便后续使用
        self.loss = config.loss
        # 创建一个用于训练过程中评估指标的对象（Evaluator），并传入类别数信息
        self.metrics_train = Evaluator(num_class=config.num_classes)
        # 创建一个用于验证过程中评估指标的对象（Evaluator），并传入类别数信息
        self.metrics_val = Evaluator(num_class=config.num_classes)

    # 前向传播方法中只使用了网络模型 (self.net) 进行预测/推断，而没有涉及其他训练相关的操
    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    # 括模型的前向传播、损失计算和评估指标的更新
    def training_step(self, batch, batch_idx):
        # 从数据批次中获取图像和相应的标签
        img, mask = batch['img'], batch['gt_semantic_seg']

        # 使用网络模型进行前向传播，获取模型对图像的预测。
        prediction = self.net(img)
        # 计算模型预测与真实标签之间的损失
        loss = self.loss(prediction, mask)

        # 如果配置中启用了辅助损失（auxiliary loss），则对预测结果进行处理
        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        # 取预测结果中概率最大的类别作为最终的预测标签
        pre_mask = pre_mask.argmax(dim=1)
        # 更新训练过程中的评估指标
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # supervision stage
        # opt = self.optimizers(use_pl_optimizer=False)
        # self.manual_backward(loss)
        # if (batch_idx + 1) % self.config.accumulate_n == 0:
        #     opt.step()
        #     opt.zero_grad()
        #
        # sch = self.lr_schedulers()
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
        #     sch.step()

        # 返回包含损失信息的字典
        return {"loss": loss}

    # 在每个训练 epoch 结束时被调用
    # 该方法用于收尾工作，包括计算并输出训练过程中的评估指标、打印信息以及记录日志
    def on_train_epoch_end(self):
        # 根据配置文件中的 log_name，选择不同的计算方式config文件夹中
        if 'QTPL' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'WHDLD' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'LoveDA' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())


        # 计算训练集上的平均OverallAccuracy(OA)
        OA = np.nanmean(self.metrics_train.OA())
        # 获取训练集上每个类别的 Intersection over Union (IoU)
        iou_per_class = self.metrics_train.Intersection_over_Union()

        # 计算精确度和召回率
        precision = np.nanmean(np.nan_to_num(self.metrics_train.Precision()))
        recall = np.nanmean(np.nan_to_num(self.metrics_train.Recall()))
        # 计算Kappa
        kappa = self.metrics_train.Kappa()

        # 将计算得到的 mIoU、F1 和 OA 存储在一个字典中
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA,
                      'Precision': precision,
                      'Recall': recall,
                      'Kappa': kappa}
        # 打印训练集上的评估指标信息
        print("  ")
        print('train evaluation:', eval_value)

        # 处理每个类别的 IoU，打印相关信息，并将训练集上的评估指标记录到日志中
        # 初始化一个空字典，用于存储每个类别的 IoU
        iou_value = {}
        # 遍历类别和对应的 IoU
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # 打印每个类别的 IoU
        print(iou_value)
        print("======================")
        print(" ")

        # 重置训练集上的评估指标，以便下一个 epoch 使用
        self.metrics_train.reset()

        # 将 mIoU、F1 和 OA 存储在一个字典中
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA, 'train_Precision': precision, 'train_Recall': recall, 'train_Kappa': kappa}
        # 将训练集上的评估指标记录到日志中
        self.log_dict(log_dict, prog_bar=True)

    # 计算验证集上的损失和评估指标
    def validation_step(self, batch, batch_idx):
        # 从验证集的批次中获取图像和相应的标签
        img, mask = batch['img'], batch['gt_semantic_seg']
        # 使用网络模型进行前向传播，获取模型对图像的预测
        prediction = self.forward(img)



        # 对预测结果进行 softmax 操作，将其转换为概率分布
        pre_mask = nn.Softmax(dim=1)(prediction)
        # 取每个像素预测概率最大的类别作为预测结果
        pre_mask = pre_mask.argmax(dim=1)
        # 遍历当前批次中的每张图像
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # 计算模型预测与真实标签之间的损失
        loss_val = self.loss(prediction, mask)
        # 返回包含验证集上损失的字典，用于记录和后续处理
        return {"loss_val": loss_val}

    # 每个验证 epoch 结束时被调用
    def on_validation_epoch_end(self):
        if 'QTPL' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])

        elif 'WHDLD' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])

        elif 'LoveDA' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])

        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())


        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        # 计算精确度和召回率，处理可能的nan
        precision = np.nanmean(np.nan_to_num(self.metrics_val.Precision()))
        recall = np.nanmean(np.nan_to_num(self.metrics_val.Recall()))

        # 计算Kappa
        kappa = self.metrics_val.Kappa()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA,
                      'Precision': precision,
                      'Recall': recall,
                      'Kappa': kappa}
        print(" ")
        print('val evaluation:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        print("======================")

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA, 'val_Precision': precision, 'val_Recall': recall, 'val_Kappa': kappa}
        self.log_dict(log_dict, prog_bar=True)

    # 配置优化器和学习率调度器
    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    # 训练的 DataLoader 对象，train_loader 是配置文件中指定的训练数据集加载器
    def train_dataloader(self):

        return self.config.train_loader

    # 验证的 DataLoader 对象。val_loader 是配置文件中指定的验证数据集加载器
    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    # 从命令行参数中获取配置文件路径
    args = get_args()
    # 将配置文件转换为配置对象
    config = py2cfg(args.config_path)
    # 设置随机种子
    seed_everything(42)

    # 配置模型保存的回调函数
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    # 配置日志记录器
    logger = CSVLogger('lightning_logs', name=config.log_name)

    # 创建模型实例
    model = Supervision_Train(config)

    # 如果指定了预训练的检查点路径，则加载预训练的模型
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    # 配置 PyTorch Lightning 的 Trainer
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='auto',
                         logger=logger)

    # 开始模型训练
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
