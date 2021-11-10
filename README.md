# Installtion & Setup
We follow the installation precess of Unbiased Teacher official repo (https://github.com/facebookresearch/unbiased-teacher)

### Prerequisites

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.

### Build Detectron2 from Source
- We find the latest(v0.6) package of Detectron2 occur the error with our code.
- Therefore, please install the matched(v0.5) version of Detectron2 as follows:

```shell
# get the Detectron2 v0.5 package
wget https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.5.zip

# unzip
unzip v0.5.zip

# install
python -m pip install -e detectron2-0.5

```


### Install other requirements
```shell
pip install -r requirements.txt
```

### Dataset download

1. Download COCO & VOC dataset

2. Organize the dataset as following:

```shell
mix-unmix/
└── datasets/
    ├── coco/
    │   ├── train2017/
    │   ├── val2017/
    │   └── annotations/
    │   	├── instances_train2017.json
    │   	└── instances_val2017.json
    ├── VOC2007
    │   ├── Annotations
    │   ├── ImageSets
    │   └── JPEGImages
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        └── JPEGImages

```

# Evaluation

- Model Weights

|  Backbone  | Protocols |         AP50  |  AP50:95      |                                       Model Weights                                        |
| :-----: | :---------: | :---: | :---: | :----------------------------------------------------------------------------------------: |
| R50-FPN |     COCO-Standard 1%       | 40.06 | 21.89 | [link](https://drive.google.com/file/d/1NxHjtz4ioFnCfRJSxskqP_zkbnWVnIeu/view?usp=sharing) |
| R50-FPN |     COCO-Additional       |  63.30 | 42.11 | [link](https://drive.google.com/file/d/1GhQlkurzdRAngdMp6Ut492TYD2AN20XB/view?usp=sharing) |
| R50-FPN |     VOC07 (VOC12)       |  78.94  | 50.22 | [link](https://drive.google.com/file/d/1HVAMThGp9SR5BpmQEBFautuF_pQlkkQW/view?usp=sharing) |
| R50-FPN |     VOC07 (VOC12 / COCO20cls)  | 80.45 | 52.31 | [link](https://drive.google.com/file/d/1Ywlnnxfi3fYwZK5jZKY7a8E7R0KP1SUs/view?usp=sharing) |
| Swin    |     COCO-Standard 0.5%    | 33.99 | 16.74 | [link](https://drive.google.com/file/d/19q73qCw1XGTWhmHrFTtr-PxbNTXnJUy0/view?usp=sharing) |


- Run Evaluation w/ R50 in COCO
```shell
python train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/mum_configs/coco.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

- Run Evaluation w/ R50 in VOC
```shell
python train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/mum_configs/voc.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

# Train
We use 4 GPUs (A6000 or V100 32GB) to achieve the paper results.   
- Train the MUM under 1% COCO-supervision (ResNet-50)
```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/mum_configs/coco.yaml \
```

- Train the MUM under VOC07 as labeled set and VOC12 as unlabeled set
```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/mum_configs/voc.yaml \
```

## Swin 
- Download ImageNet pretrained weight of swin-t in [link](https://drive.google.com/file/d/1j95KPUoVl1PK49yxpQOvigKHcl2eTt5B/view?usp=sharing)
- mv pretrained weight to weights folder
```shell
mv swin_tiny_patch4_window7_224.pth weights/
```

- Run Evaluation w/ Swin in COCO
```shell
python train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/mum_configs/coco_swin.yaml \
      MODEL.WEIGHTS <your weight>.pth
      
```

- Train under 0.5% COCO-supervision
```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/mum_configs/coco_swin.yaml \
```

## Mix/UnMix code block

### Mixing code block
- Generate mix mask 
```shell
mask = torch.argsort(torch.rand(bs // ng, ng, nt, nt), dim=1).cuda()
img_mask = mask.view(bs // ng, ng, 1, nt, nt)
img_mask = img_mask.repeat_interleave(3, dim=2)
img_mask = img_mask.repeat_interleave(h // nt, dim=3)
img_mask = img_mask.repeat_interleave(w // nt, dim=4)
```

- Mixing image tiles
```shell
img_tiled = images.tensor.view(bs // ng, ng, c, h, w)
img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
img_tiled = img_tiled.view(bs, c, h, w)
```

### Unmixing  code block

- Generate inverse mask to unmix
```shell
inv_mask = torch.argsort(mask, dim=1).cuda()
feat_mask = inv_mask.view(bs//ng,ng,1,nt,nt)
feat_mask = feat_mask.repeat_interleave(c,dim=2)
feat_mask = feat_mask.repeat_interleave(h//nt, dim=3)
feat_mask = feat_mask.repeat_interleave(w//nt, dim=4)
```

- Unmixing feature tiles
```shell
feat_tiled = feat.view(bs//ng,ng,c,h,w)
feat_tiled = torch.gather(feat_tiled, dim=1, index=feat_mask)
feat_tiled = feat_tiled.view(bs,c,h,w)
```