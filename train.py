import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader_sequence import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from tqdm import tqdm

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=15)
    parser.add_argument('--sequence_length', help='Number of frames in each temporal sequence', type=int, default=3)
    parser.add_argument('--use_temporal', help='Use temporal model', action='store_true')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO.')

        dataset_train = CocoDataset(
            parser.coco_path,
            set_name='imagenet_vid_train_15frames',
            sequence_length=parser.sequence_length,
            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
            only_train_frames=False
        )
        dataset_val = CocoDataset(
            parser.coco_path,
            set_name='imagenet_vid_val',
            sequence_length=parser.sequence_length,
            transform=transforms.Compose([Normalizer(), Resizer()]),
            only_train_frames=False
        )
    else:
        raise ValueError('Dataset type not understood (must be coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=5, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=2, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, use_temporal=parser.use_temporal)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, use_temporal=parser.use_temporal)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, use_temporal=parser.use_temporal)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, use_temporal=parser.use_temporal)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, use_temporal=parser.use_temporal)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    try:
        for epoch_num in range(parser.epochs):

            retinanet.train()
            retinanet.module.freeze_bn()

            epoch_loss = []

            progress_bar = tqdm(dataloader_train, desc=f'Epoch {epoch_num + 1}/{parser.epochs}', unit='batch')

            for iter_num, data in enumerate(progress_bar):
                    optimizer.zero_grad()

                    # Handle temporal sequences: `img` shape is [B, T, C, H, W]
                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                        
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                    if bool(loss == 0):
                        continue

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                    optimizer.step()

                    loss_hist.append(float(loss))

                    epoch_loss.append(float(loss))

                    progress_bar.set_postfix({
                        'Classification loss': '{:.5f}'.format(float(classification_loss)),
                        'Regression loss': '{:.5f}'.format(float(regression_loss)),
                        'Running loss': '{:.5f}'.format(np.mean(loss_hist))
                    })

                    del classification_loss
                    del regression_loss

            save_model_path = '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)
            torch.save(retinanet.module, save_model_path)
            if parser.dataset == 'coco':
                print('Evaluating dataset')
                coco_eval.evaluate_coco_sequence(dataset_val, retinanet, model_path=save_model_path)  # Use the updated temporal `coco_eval`

            scheduler.step(np.mean(epoch_loss))


    except KeyboardInterrupt:
        print('Training interrupted. Saving model...')
        torch.save(retinanet.module, 'interrupted_model.pt')

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
