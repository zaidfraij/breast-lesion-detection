from __future__ import print_function

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import pandas as pd
from helper import load_json


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def process_image(data, retinanet, score_threshold, max_detections):
    scale = data['scale']

    img = data['img']

    # run network
    if torch.cuda.is_available():
        scores, labels, boxes = retinanet(img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
    else:
        scores, labels, boxes = retinanet(img.permute(2, 0, 1).float().unsqueeze(dim=0))
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    boxes  = boxes.cpu().numpy()

    # correct boxes for image scale
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores > score_threshold)[0]
    if indices.shape[0] > 0:
        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
    else:
        image_detections = np.zeros((0, 6))

    return image_detections


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=1, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]
            image_detections = process_image(data, retinanet, score_threshold, max_detections)

            # copy detections to all_detections
            for label in range(dataset.num_classes()):
                all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=1,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations

    all_detections     = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        print(precision)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    metrics_list= []
    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        #print('{}: {}'.format(label_name, average_precisions[label][0]))
        print(label_name)
        print("Average Precision (AP): ", average_precisions[label][0])
        print(f"Precision (AP{iou_threshold*100}): ", precision[-1])
        print("Recall: ", recall[-1])
        metric_dict= {"label": label_name, "AP": average_precisions[label][0], "Precision": precision[-1], "Recall": recall[-1]}
        metrics_list.append(metric_dict)
        if save_path is not None:
            plt.plot(recall, precision)
            # naming the x axis 
            plt.xlabel('Recall') 
            # naming the y axis 
            plt.ylabel('Precision') 

            # giving a title to my graph 
            plt.title('Precision Recall curve') 

            # function to show the plot
            plt.savefig(save_path + '/' + label_name + '_precision_recall.jpg')



    return metrics_list

def evaluate_coco_metrics(generator, retinanet, buv_json_file, model_path, score_threshold=0.05, max_detections=1):
    """
    Evaluate a given dataset using a given retinanet, returning COCO metrics.
    """
    # Prepare COCO-style ground truth annotations
    ground_truth = []
    json_dict = load_json(buv_json_file)
    dict_keys = list(json_dict.keys())
    dict_image_data = json_dict[dict_keys[2]]
    dict_annotation_data = json_dict[dict_keys[3]]
    images_df = pd.DataFrame(dict_image_data)
    annotation_df = pd.DataFrame(dict_annotation_data)
    print('Preparing ground truth annotations...')
    for i in tqdm(range(len(generator))):
        relative_path = os.path.relpath(generator.image_names[i], start='rawframes')
        file_name = '/'.join(relative_path.split('\\')[-3:])
        image_id = images_df[images_df['file_name'] == file_name]['id'].values[0]
        annotation = annotation_df[annotation_df['image_id'] == image_id].to_dict(orient='records')[0]
        annotation['category_id'] = 0
        bbox = generator.load_annotations(i)[0][:4]
        annotation['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])] # [x_min, y_min, x_max, y_max]
        ground_truth.append(annotation)

    # Prepare COCO-style predictions
    print('Preparing predictions...')
    predictions = []
    retinanet.eval()
    with torch.no_grad():
        for i in tqdm(range(len(generator))):
            data = generator[i]
            relative_path = os.path.relpath(generator.image_names[i], start='rawframes')
            file_name = '/'.join(relative_path.split('\\')[-3:])
            image_id = images_df[images_df['file_name'] == file_name]['id'].values[0]
            detection = process_image(data, retinanet, score_threshold, max_detections)
            if len(detection) == 0:
                print('Empty Detection: ', detection)
                det = [0, 0, 0, 0, 0, 0]
            else:
                det = detection[0]

            predictions.append({
                "image_id": int(image_id),
                "bbox": [float(det[0]), float(det[1]), float(det[2]), float(det[3])], # [x_min, y_min, x_max, y_max]
                "score": float(det[4]),
                "category_id": int(det[5])
            })

    # Load ground truth and predictions into COCO and COCOeval
    coco_gt = COCO()
    bbox_output_file = f'{model_path}_bbox_results.json'

    try:
        with open(bbox_output_file, 'w') as f:
            json.dump(predictions, f, indent=4)
    except Exception as e:
        print(f'Error writing to {bbox_output_file}: {e}')

    coco_gt.dataset = {
        "images": dict_image_data,
        "categories": [{"id": i, "name": generator.label_to_name(i)} for i in range(generator.num_classes())],
        "annotations": ground_truth
    }
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Collect metrics in a dictionary
    metrics = {
        "AP@[IoU=0.50:0.95]": coco_eval.stats[0],
        "AP@[IoU=0.50]": coco_eval.stats[1],
        "AP@[IoU=0.75]": coco_eval.stats[2],
        "AP@[small]": coco_eval.stats[3],
        "AP@[medium]": coco_eval.stats[4],
        "AP@[large]": coco_eval.stats[5],
        "AR@[max=1]": coco_eval.stats[6],
        "AR@[max=10]": coco_eval.stats[7],
        "AR@[max=100]": coco_eval.stats[8],
        "AR@[small]": coco_eval.stats[9],
        "AR@[medium]": coco_eval.stats[10],
        "AR@[large]": coco_eval.stats[11]
    }

    return metrics

