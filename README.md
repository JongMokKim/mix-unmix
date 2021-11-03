# Installtion & Setup
Follow the installtion process of Unbiased Teacher official repo (https://github.com/facebookresearch/unbiased-teacher)

<details>
<summary>Install guideline</summary>
<div markdown="1">

### Prerequisites

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.

### Install PyTorch in Conda env

```shell
# create conda env
conda create -n detectron2 python=3.6
# activate the enviorment
conda activate detectron2
# install PyTorch >=1.5 with GPU
conda install pytorch torchvision -c pytorch
```

### Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

### Install Timm for Swin Backbone
```shell
# install timm for SwinTransformer
pip install timm
```

### Dataset download

1. Download COCO dataset

```shell
# download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

2. Organize the dataset as following:

```shell
unbiased_teacher/
└── datasets/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

</div>
</details>

# Evaluation

- Model Weights

|  Backbone  | Protocols |         AP50  |  AP50:95      |                                       Model Weights                                        |
| :-----: | :---------: | :---: | :---: | :----------------------------------------------------------------------------------------: |
| R50-FPN |     COCO-Standard 1%       | 40.05 | 21.887 | [link](https://drive.google.com/file/d/1NxHjtz4ioFnCfRJSxskqP_zkbnWVnIeu/view?usp=sharing) |
| R50-FPN |     COCO-Additional       |  63.31 | 42.12 | [link](https://drive.google.com/file/d/1GhQlkurzdRAngdMp6Ut492TYD2AN20XB/view?usp=sharing) |
| R50-FPN |     VOC07 (VOC12)       |  78.94  | 50.22 | [link](https://drive.google.com/file/d/1HVAMThGp9SR5BpmQEBFautuF_pQlkkQW/view?usp=sharing) |
| R50-FPN |     VOC07 (VOC12 / COCO20cls)  | 80.46 | 52.31 | [link](https://drive.google.com/file/d/1Ywlnnxfi3fYwZK5jZKY7a8E7R0KP1SUs/view?usp=sharing) |
| Swin    |     COCO-Standard 0.5%    | 33.98 | 16.74 | [link](https://drive.google.com/file/d/19q73qCw1XGTWhmHrFTtr-PxbNTXnJUy0/view?usp=sharing) |


- Run Evaluation w/ R50 in COCO
```shell
python train_net.py \
      --eval-only \
      --num-gpus 4 \
      --config configs/mum_configs/coco.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

- Run Evaluation w/ R50 in VOC
```shell
python train_net.py \
      --eval-only \
      --num-gpus 4 \
      --config configs/mum_configs/voc.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

- Run Evaluation w/ Swin in COCO
```shell
python train_net.py \
      --eval-only \
      --num-gpus 4 \
      --config configs/mum_configs/coco_swin.yaml \
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

- Train the MUM under 0.5% COCO-supervision (Swin)
```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/mum_configs/coco_swin.yaml \
```

## Mix/UnMix code block

- Mix images tiles
```shell
# Generate tile mask (BS//NT, NT, TH, TW)
mask = torch.argsort(torch.rand(bs // nt, nt, ts, ts), dim=1).cuda()

# Repeat tile mask to make image shape (BS//NT, NT, 3, H, W)
img_mask = mask.view(bs // nt, nt, 1, ts, ts)
img_mask = img_mask.repeat_interleave(3, dim=2)
img_mask = img_mask.repeat_interleave(h // ts, dim=3)
img_mask = img_mask.repeat_interleave(w // ts, dim=4)

# Tiling image
img_tiled = images.tensor.view(bs // nt, nt, c, h, w)
img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
img_tiled = img_tiled.view(bs, c, h, w)

```

- Unmix Feature tiles 
```shell

bs, c, h, w = feat.shape

# Make feature shape multiple of ts 
h_ = h//ts * ts
w_ = w//ts * ts

if h_ == 0:
    h_ = ts
if w_ == 0:
    w_ = ts

if h != h_ or w != w_:
    feat = torch.nn.functional.interpolate(feat, size=(h_, w_), mode='bilinear')

# Generate inverse mask to untile
inv_mask = torch.argsort(mask, dim=1).cuda()

feat_tiled = feat.view(bs//nt,nt,c,h_,w_)
feat_mask = inv_mask.view(bs//nt,nt,1,ts,ts)
feat_mask = feat_mask.repeat_interleave(c,dim=2)
feat_mask = feat_mask.repeat_interleave(h_//ts, dim=3)
feat_mask = feat_mask.repeat_interleave(w_//ts, dim=4)

# Untile feature
feat_tiled = torch.gather(feat_tiled, dim=1, index=feat_mask)
feat_tiled = feat_tiled.view(bs,c,h_,w_)
if h != h_:
    feat_tiled = torch.nn.functional.interpolate(feat_tiled, size=(h, w), mode='bilinear')



```
