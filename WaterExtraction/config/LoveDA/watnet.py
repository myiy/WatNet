from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
# from geoseg.models.ABCNet import ABCNet
# from geoseg.models.BANet import BANet
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.UNet import Unet
from geoseg.models.PSPNet import PSPNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from geoseg.models.convlsrnet import Model


# training hparam
max_epoch = 40
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 1e-4
weight_decay = 0.01
backbone_lr = 1e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "convlsnet-r50-512crop-ms-epoch40-rep"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "last"
log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = Model(num_classes=num_classes)
# net = ABCNet(num_classes=num_classes)
# net = BANet(num_classes=num_classes)
# net = Unet(num_classes=num_classes)
# net = PSPNet(class_num=num_classes)
# net = MUNet_704(in_channels=3,out_channels=1)
# define the loss
loss = EdgeLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader

def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/Train')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
