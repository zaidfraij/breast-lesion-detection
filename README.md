# Breast Lesion Detection Experiments on Breast Ultra-Sound Video Datasets
The goal of this project is to develop, evaluate, and experiment with object detection models for automatically identifying and localizing breast lesions in ultrasound videos.

Main Focus:

    - Develop a deep learning-based object detection model for identifying and localizing breast lesions in ultrasound videos.
    - Explore temporal coherence between video frames to improve detection accuracy.
    - Evaluate the model’s performance on publicly available datasets, comparing it to baseline methods.

## Installation

1) Clone this repo

2) Create conda environment for this project (Recommended)

3) Install PyTorch and torchvision (following instructions [here](https://pytorch.org/))

    For example: 
    
    ```
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

5) Install the required packages:

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

We use RetinaNet with pre-trained ResNet as backbone (see [Repo](https://github.com/yhenon/pytorch-retinanet))

You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

## Training

The network can be trained using the `train.py` script through a COCO dataloader.

```
python train.py --dataset coco --coco_path "..\Miccai 2022 BUV Dataset" --depth 50
```

## Validation

Run `coco_validation.py` to validate the code on the COCO dataset. With the above model, run:

`python coco_validation.py --coco_path ~/path/to/coco --model_path /path/to/model/coco_resnet_50_map_0_335_state_dict.pt`


This produces the following results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.597
```

For CSV Datasets (more info on those below), run the following script to validate:

## Visualization

To visualize the network detection, use `visualize.py`:

```
python visualize.py --dataset coco --coco_path ../coco --model <path/to/model.pt>
```
This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:

```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```
