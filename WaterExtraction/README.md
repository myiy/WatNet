
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── GeoSeg (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── qtpl
│   │   ├── qtpl_train (original)
│   │   ├── qtpl_val (original)
│   │   ├── qtpl_test (original)
│   │   ├── qtpl_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
train：test：val=7:2:1   
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

## Pretrained Weights of Backbones

[Baidu Disk](https://pan.baidu.com/s/1foJkxeUZwVi5SnKNpn6hfg) : 1234 

[Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)

## Data Preprocessing

Download the datasets from the official website and split them yourself.


**QTPL**
```
生成训练集
python GeoSeg/tools/qtpl_patch_split.py \
--input-img-dir "data/QTPL/Train/Images" \
--input-mask-dir "data/QTPL/Train/Labels" \
--output-img-dir "data/QTPL/Train/images_png" \
--output-mask-dir "data/QTPL/Train/masks_png" \
--mode 'train' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256
```


```
生成用于可视化的masks_512_rgb
python tools/qtpl_patch_split.py \
--input-img-dir "data/QTPL/Val/Images" \
--input-mask-dir "data/QTPL/Val/Labels" \
--output-img-dir "data/QTPL/Val/images_png" \
--output-mask-dir "data/QTPL/Val/masks_png" \
--mode 'val' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256
```

**WHDLD**
```
python tools/whdld_patch_split.py \
--input-img-dir "data/WHDLD/Train/Images" \
--input-mask-dir "data/WHDLD/Train/Labels" \
--output-img-dir "data/WHDLD/Train/images_png" \
--output-mask-dir "data/WHDLD/Train/masks_png" \
--mode 'train' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256
```


```
生成用于可视化的masks_512_rgb
python tools/whdld_patch_split.py \
--input-img-dir "data/WHDLD/Val/Images" \
--input-mask-dir "data/WHDLD/Val/Labels" \
--output-img-dir "data/WHDLD/Val/images_png" \
--output-mask-dir "data/WHDLD/Val/masks_png" \
--mode 'val' --split-size-h 256 --split-size-w 256 \
--stride-h 256 --stride-w 256
```


**LoveDA**
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```



至于验证集，你可以从训练集中选择一些图像来构建它

## Training

"-c" means the path of the config, use different **config** to train different models.

**QTPL**
```
python train_supervision.py -c config/QTPL/unetformer.py
python train_supervision.py -c config/QTPL/watnet.py
```


**WHDLD**
```
python train_supervision.py -c config/WHDLD/watnet.py
```


**LoveDA**
```
python train_supervision.py -c config/loveda/unetformer.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**QTPL**
```
python qtpl_test.py -c config/qtpl/watnet.py -o fig_results/qtpl/WatNet --rgb -t lr
```

**LoveDA**
```
python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/unetformer.py -o fig_results/loveda/unetformer --rgb -t d4
```






