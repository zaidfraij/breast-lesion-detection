# Breast Lesion Detection Experiments on Breast Ultra-Sound Video Datasets Using Retinanet + Temporal Attention
The goal of this project is to develop, evaluate, and experiment with Retinanet with simple temporal attention layers for automatically identifying and localizing breast lesions in ultrasound videos.

Main Focus:

```
    - Develop a deep learning-based object detection model for identifying and localizing breast lesions in ultrasound videos.
    - Explore adding temporal attention layer to caputre temporal coherance between video frames to improve detection accuracy.
    - Evaluate the model’s performance on publicly available datasets, comparing it to baseline models.
```


## Installation

1) Clone this repo

2) Create conda environment for this project (Recommended)

3) Install PyTorch and torchvision (following instructions [here](https://pytorch.org/))

For example: 

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```


4) Install the required packages:

```
pip install -r requirements.txt
```

## Dataset
We use the annotated ultra-sound breast videos from https://arxiv.org/pdf/2207.00141

```
code_root/
Miccai 2022 BUV Dataset/
      ├── rawframes/
      ├── annotations/
          ├── instances_imagenet_vid_train_15frames.json
          └── instances_imagenet_vid_val.json
```

## Baseline

We use RetinaNet with pre-trained ResNet as backbone (GitHub GitHub - yhenon/pytorch-retinanet: Pytorch implementation of…)

You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

## Training

The network can be trained using the `train.py` script through a COCO dataloader.

```
python train.py --dataset coco --coco_path "..\Miccai 2022 BUV Dataset" --depth 50
```

## Trained Model
[GoogleDrive](https://drive.google.com/file/d/1hHYoL8R7wcCeGQVuzD41iTgZRs4YpRXQ/view?usp=sharing)

## Validation

Run `coco_validation.py` to validate the code on the COCO dataset. With the above model, run:

`python coco_validation.py --coco_path /path/to/Miccai 2022 BUV Dataset --model_path /path/to/model/temporal_retinanet_resnet50_buv_weights.pt`


This produces the following results:

```
"AP@[IoU=0.50:0.95]": 0.18678210301746864,
"AP@[IoU=0.50]": 0.37237278898779586,
"AP@[IoU=0.75]": 0.16718358991856494,
"AP@[small]": -1.0,
"AP@[medium]": 0.08179865576647725,
"AP@[large]": 0.1893966697034638,
"AR@[max=1]": 0.4591271628622534,
"AR@[max=10]": 0.5516916167862869,
"AR@[max=100]": 0.5572792917317064,
"AR@[small]": -1.0,
"AR@[medium]": 0.3876760563380282,
"AR@[large]": 0.5604575783229951
```

## Notes
The code of this repository is built on [https://github.com/jhl-Det/CVA-Net](https://github.com/yhenon/pytorch-retinanet).
