from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import json
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image
from future.utils import raise_from
from tqdm import tqdm



class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None, sequence_length=3, only_train_frames=True):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Number of frames in each sequence.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.sequence_length = sequence_length

        set_json_path = os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')
        self.coco = COCO(set_json_path)

        with open(set_json_path, 'r') as file:
            json_dict = json.load(file)
        dict_keys = list(json_dict.keys())
        dict_video_data = json_dict[dict_keys[1]]
        dict_image_data = json_dict[dict_keys[2]]
        images_df = pd.DataFrame(dict_image_data)

        if only_train_frames:
            images_df = images_df[images_df['is_vid_train_frame'] == True]
            self.image_ids = list(images_df.id)
        else:
            # Exclude the last [sequence_length] frames of each video
            self.image_ids = []
            for video_id in images_df['video_id'].unique():
                video_frames = images_df[images_df['video_id'] == video_id]
                if len(video_frames) > sequence_length:
                    self.image_ids.extend(video_frames.iloc[:-sequence_length].id)
                else:
                    self.image_ids.extend(video_frames.id)

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns a sequence of images and corresponding annotations.
        """
        sequence_imgs = []
        sequence_annots = []

        # Load a sequence of images and annotations
        for offset in range(self.sequence_length):

            current_image_id = self.image_ids[int(idx)] + int(offset)
            img = self.load_image(current_image_id)
            annot = self.load_annotations(current_image_id)
            sequence_imgs.append(img)
            sequence_annots.append(annot)

        sequence_imgs = np.stack(sequence_imgs)  # Shape: [T, H, W, C]
        sequence_annots = np.array(sequence_annots, dtype=object)  # Annotations per frame

        sample = {'img': sequence_imgs, 'annot': sequence_annots}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        path       = os.path.join(self.root_dir, 'rawframes', image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_id):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 2

def collater(data):
    """
    Collate function for batching temporal sequences of images and annotations.
    """
    imgs = [s['img'] for s in data]  # List of [T, H, W, C]
    annots = [s['annot'] for s in data]  # List of [T, M, 5]
    batch_size = len(imgs)
    time_steps = imgs[0].shape[0]  # Number of frames per sequence

    # Determine the max spatial dimensions across all frames in the batch
    max_height = max(img.shape[0] for seq in imgs for img in seq)  # Max height
    max_width = max(img.shape[1] for seq in imgs for img in seq)  # Max width
    channels = imgs[0].shape[3]  # Number of channels (e.g., 3 for RGB)
    # Initialize tensors for padded images and annotations
    padded_imgs = torch.zeros(batch_size, time_steps, channels, max_height, max_width, dtype=torch.float32)
    max_num_annots = max(max(annot.shape[1] for annot in seq) for seq in annots)

    if max_num_annots > 0:
        padded_annots = torch.ones(batch_size, time_steps, max_num_annots, 5, dtype=torch.float32) * -1
    else:
        padded_annots = torch.ones(batch_size, time_steps, 1, 5, dtype=torch.float32) * -1

    # Pad images and annotations
    for b in range(batch_size):
        for t in range(time_steps):
            img = imgs[b][t]  # Single frame: [H, W, C]
            img = img.clone().detach().permute(2, 0, 1)  # Convert to [C, H, W]
            padded_imgs[b, t, :, :img.shape[1], :img.shape[2]] = img

            annot = annots[b][t]  # Annotations for this frame
            if annot.shape[0] > 0:
                padded_annots[b, t, :annot.shape[0], :] = annot.clone().detach()

    return {'img': padded_imgs, 'annot': padded_annots}


class Normalizer(object):
    """Normalize a sequence of frames."""

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        images, annots = sample['img'], sample['annot']  # `images`: [T, H, W, C], `annots`: [T, M, 5]

        time_steps = images.shape[0]
        normalized_images = []

        for t in range(time_steps):
            image = images[t]

            # Remove alpha channel if it exists
            if image.shape[2] == 4:
                image = image[:, :, :3]

            # Normalize image
            image = (image.astype(np.float32) - self.mean) / self.std
            normalized_images.append(image)

        normalized_images = np.stack(normalized_images)  # Shape: [T, H, W, C]
        return {'img': normalized_images, 'annot': annots}


class Resizer(object):
    """Resize a sequence of frames and annotations to the desired dimensions."""

    def __call__(self, sample, min_side=608, max_side=1024):
        images, annots = sample['img'], sample['annot']  # `images`: [T, H, W, C], `annots`: [T, M, 5]
        time_steps, original_height, original_width, channels = images.shape

        new_images_list = []
        new_annots_list = []

        max_annots = max(annot.shape[0] for annot in annots)  # Find the maximum number of annotations across frames
        new_annots_array = np.ones((time_steps, max_annots, 5), dtype=np.float32) * -1
        for t in range(time_steps):
            image = images[t]
            annot = annots[t]
            new_image, new_annots, scale = self.resize_one_image(image, annot, min_side, max_side)
            new_images_list.append(new_image)
            if annot.shape[0] > 0:
                new_annots_array[t, :annot.shape[0], :] = new_annots
        
        new_images_array = np.array(new_images_list)

        return {
            'img': torch.from_numpy(new_images_array),  # Shape: [T, H', W', C]
            'annot': torch.from_numpy(new_annots_array),  # Shape: [T, M, 5]
            'scale': scale,  # Single scale value used for all frames
        }
    
    def resize_one_image(self, image, annots, min_side, max_side):
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return new_image, annots, scale


class Augmenter(object):
    """Apply random horizontal flipping to a sequence of frames and annotations."""

    def __call__(self, sample, flip_x=0.5):
        images, annots = sample['img'], sample['annot']  # `images`: [T, H, W, C], `annots`: [T, M, 5]

        time_steps = images.shape[0]
        augmented_images = []
        augmented_annots = []
        flip_bool = np.random.rand() < flip_x

        for t in range(time_steps):
            image = images[t]
            annot = annots[t]

            if flip_bool:
                image = image[:, ::-1, :]  # Flip horizontally
                rows, cols, channels = image.shape

                x1 = annot[:, 0].copy()
                x2 = annot[:, 2].copy()

                annot[:, 0] = cols - x2
                annot[:, 2] = cols - x1

            augmented_images.append(image)
            augmented_annots.append(annot)

        augmented_images = np.stack(augmented_images)  # Shape: [T, H, W, C]
        augmented_annots = np.array(augmented_annots, dtype=object)  # Shape: [T, M, 5]

        return {'img': augmented_images, 'annot': augmented_annots}

class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        """
        Args:
            data_source (Dataset): Dataset instance.
            batch_size (int): Number of samples per batch.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        #random.shuffle(self.groups)  # Shuffle groups for randomness
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        """
        Groups images based on their aspect ratio for efficient batching.
        For temporal sequences, the aspect ratio of the first frame in the sequence is used.
        """
        # Determine the order of images based on aspect ratio of the first frame
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))  # Uses first frame's aspect ratio

        # Divide into groups, one group = one batch
        return [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[self.rank // self.num_parts:self.total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch