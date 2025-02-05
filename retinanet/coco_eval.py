from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import auc
from helper import load_json
import json
import torch
import os
import numpy as np



def evaluate_coco(dataset, model, threshold=0.05, model_path=None):
    
    model.eval()
    
    with torch.no_grad():

        bbox_output_file = f'{model_path}_bbox_results.json'
        if os.path.exists(bbox_output_file):
            print(f'Loading existing predictions from {bbox_output_file}...')
            coco_true = dataset.coco
            image_ids = [dataset.image_ids[index] for index in range(len(dataset))]
            coco_pred = coco_true.loadRes(bbox_output_file)
            print(coco_pred)
        else:
            # start collecting results
            results = []
            image_ids = []

            for index in range(len(dataset)):
                data = dataset[index]
                scale = data['scale']

                # run network
                if torch.cuda.is_available():
                    scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                else:
                    scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
                scores = scores.cpu()
                labels = labels.cpu()
                boxes  = boxes.cpu()

                # correct boxes for image scale
                boxes /= scale

                # Get the ground truth category ID for the current image
                gt_annotations = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=dataset.image_ids[index]))
                if len(gt_annotations) > 0:
                    gt_category_id = gt_annotations[0]['category_id']
                else:
                    gt_category_id = 1  # Default category ID if no ground truth annotations are found

                if boxes.shape[0] > 0:
                    # change to (x, y, w, h) (MS COCO standard)
                    boxes[:, 2] -= boxes[:, 0]
                    boxes[:, 3] -= boxes[:, 1]

                    # compute predicted labels and scores
                    #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # scores are sorted, so we can break
                        if score < threshold:
                            break

                        # append detection for each positively labeled class
                        image_result = {
                            'image_id'    : dataset.image_ids[index],
                            'category_id' : gt_category_id,
                            'score'       : float(score),
                            'bbox'        : box.tolist(),
                        }

                        # append detection to results
                        results.append(image_result)

                # append image to list of processed images
                image_ids.append(dataset.image_ids[index])

                # print progress
                print('{}/{}'.format(index, len(dataset)), end='\r')

            if not len(results):
                return

            # write output
            json.dump(results, open(bbox_output_file, 'w'), indent=4)

            # load results in COCO evaluation tool
            coco_true = dataset.coco
            coco_pred = coco_true.loadRes(bbox_output_file)

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

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

        model.train()



        return metrics

def evaluate_coco_sequence(dataset, model, threshold=0.05, model_path=None):
    """
    Evaluate the model on the COCO dataset for temporal sequences.
    Args:
        dataset: COCO dataset instance.
        model: Trained model to evaluate.
        threshold: Score threshold for filtering predictions.
    """
    model.eval()

    with torch.no_grad():

        bbox_output_file = f'{model_path}_bbox_results.json'
        if os.path.exists(bbox_output_file):
            print(f'Loading existing predictions from {bbox_output_file}...')
            coco_true = dataset.coco
            image_ids = [dataset.image_ids[index] for index in range(len(dataset))]
            coco_pred = coco_true.loadRes(bbox_output_file)
            print(coco_pred)

        else:
            # Start collecting results
            results = []
            image_ids = []

            for index in range(len(dataset)):
                data = dataset[index]  # Sample includes a sequence of frames
                scale = data['scale']  # Scale for each frame in the sequence
                imgs = data['img']  # Shape: [T, C, H, W]
                imgs = imgs.permute(0, 3, 1, 2)  # Convert to [T, C, H, W]

                if torch.cuda.is_available():
                    imgs = imgs.cuda().float().unsqueeze(dim=0)  # Add batch dimension: [1, T, C, H, W]
                else:
                    imgs = imgs.float().unsqueeze(dim=0)

                # Run network
                scores, labels, boxes = model(imgs)  # Outputs for all frames
                scores = scores.cpu()
                labels = labels.cpu()
                boxes = boxes.cpu()
                boxes /= scale

                # Get the ground truth category ID for the current image
                gt_annotations = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=dataset.image_ids[index]))
                if len(gt_annotations) > 0:
                    gt_category_id = gt_annotations[0]['category_id']
                else:
                    gt_category_id = 1  # Default category ID if no ground truth annotations are found

                if boxes.shape[0] > 0:
                    # Convert to (x, y, w, h) (MS COCO standard)
                    boxes[:, 2] -= boxes[:, 0]
                    boxes[:, 3] -= boxes[:, 1]

                    # Process each detection
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # Filter out low-confidence predictions
                        if score < threshold:
                            break

                        #category_id = dataset.label_to_coco_label(label)
                        category_id = gt_category_id

                        # Create result for COCO evaluation
                        #print(dataset.image_ids[index])
                        image_result = {
                            'image_id': dataset.image_ids[index],
                            'category_id': category_id,
                            'score': score,
                            'bbox': box.tolist(),
                        }

                        results.append(image_result)

                # Append sequence ID to processed list
                image_ids.append(dataset.image_ids[index])

                # Print progress
                print('{}/{}'.format(index + 1, len(dataset)), end='\r')

            if not len(results):
                return

            # Write output
            json.dump(results, open(bbox_output_file, 'w'), indent=4)

            # Load results in COCO evaluation tool
            coco_true = dataset.coco
            coco_pred = coco_true.loadRes(bbox_output_file)
        
        # Run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

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

        model.train()

        return metrics

def evaluate_froc(generator, retinanet, buv_json_file, model_path, score_thresholds, max_detections=100):
    """
    Evaluate FROC metrics.
    """

    # Prepare COCO-style ground truth annotations
    ground_truth = []
    json_dict = load_json(buv_json_file)
    dict_keys = list(json_dict.keys())
    dict_image_data = json_dict[dict_keys[2]]
    dict_annotation_data = json_dict[dict_keys[3]]
    images_df = pd.DataFrame(dict_image_data)
    annotation_df = pd.DataFrame(dict_annotation_data)
    num_images = len(generator)
    print('Preparing ground truth annotations...')
    for i in tqdm(range(num_images)):
        image_id = generator.image_ids[i]
        image_name =  images_df[images_df['id'] == image_id]['file_name'].values[0]
        annotations = annotation_df[annotation_df['image_id'] == image_id].to_dict(orient='records')
        for annotation in annotations:
            #bbox = generator.load_annotations(i)[0][:4]
            #annotation['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]  # [x_min, y_min, x_max, y_max]
            ground_truth.append(annotation)

    # Prepare predictions
    bbox_output_file = f'{model_path}_bbox_results.json'
    if os.path.exists(bbox_output_file):
        print(f'Loading existing predictions from {bbox_output_file}...')
        with open(bbox_output_file, 'r') as f:
            predictions = json.load(f)
    else:
        print('Preparing predictions...')
        predictions = []
        retinanet.eval()
        with torch.no_grad():
            for i in tqdm(range(num_images)):
                data = generator[i]
                relative_path = os.path.relpath(generator.image_names[i], start='rawframes')
                file_name = '/'.join(relative_path.split('\\')[-3:])
                image_id = images_df[images_df['file_name'] == file_name]['id'].values[0]
                detection = process_image(data, retinanet, score_threshold=0.0, max_detections=max_detections)  # No score threshold yet
                for det in detection:
                    predictions.append({
                        "image_id": int(image_id),
                        "bbox": [float(det[0]), float(det[1]), float(det[2]), float(det[3])],  # [x_min, y_min, x_max, y_max]
                        "score": float(det[4]),
                        "category_id": int(det[5])
                    })

        try:
            with open(bbox_output_file, 'w') as f:
                json.dump(predictions, f, indent=4)
        except Exception as e:
            print(f'Error writing to {bbox_output_file}: {e}')

    # FROC Calculation
    print('Calculating FROC...')
    iou_thresholds = [0.5, 0.75]
    froc_results = {}

    for iou_threshold in iou_thresholds:
        tprs = []
        fppis = []
        num_ground_truth = len(ground_truth)

        for threshold in score_thresholds:
            # Filter predictions based on the threshold
            filtered_predictions = [pred for pred in predictions if pred['score'] >= threshold]

            # Count true positives and false positives
            true_positives = 0
            false_positives = 0
            used_gt_ids = set()

            for pred in filtered_predictions:
                pred_bbox = pred['bbox']
                matched = False
                for gt in ground_truth:
                    if gt['image_id'] == pred['image_id'] and gt['id'] not in used_gt_ids:
                        iou = compute_iou(pred_bbox, gt['bbox'])
                        if iou >= iou_threshold:  # IoU threshold for TP
                            true_positives += 1
                            used_gt_ids.add(gt['id'])
                            matched = True
                            break
                if not matched:
                    false_positives += 1

            # Calculate TPR and FPPI
            tpr = true_positives / num_ground_truth if num_ground_truth > 0 else 0
            fppi = false_positives / num_images if num_images > 0 else 0

            tprs.append(tpr)
            fppis.append(fppi)

        froc_results[iou_threshold] = (fppis, tprs)

    # Plot FROC curve
    plt.figure(figsize=(10, 6))
    for iou_threshold, (fppis, tprs) in froc_results.items():
        plt.plot(fppis, tprs, marker='o', label=f'IoU={iou_threshold}')

        ## Annotate score thresholds
        for i, threshold in enumerate(score_thresholds):
            plt.text(fppis[i], tprs[i], f"{threshold:.2f}", fontsize=8, ha='left', va='bottom')

    plt.xlabel('False Positives Per Image (FPPI)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('FROC Curve')
    plt.legend()
    plt.grid()
    plt.show()

    # Compute AUC of FROC for each IoU threshold
    froc_aucs = {iou_threshold: auc(fppis, tprs) for iou_threshold, (fppis, tprs) in froc_results.items()}
    for iou_threshold, froc_auc in froc_aucs.items():
        print(f"FROC AUC (IoU={iou_threshold}): {froc_auc:.4f}")

    return froc_aucs

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

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: List or array [x_min, y_min, x_max, y_max] for the first box.
        box2: List or array [x_min, y_min, x_max, y_max] for the second box.

    Returns:
        iou: IoU value between 0 and 1.
    """
    # Calculate intersection coordinates
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Calculate intersection area
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Calculate areas of the individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # Compute IoU
    iou = intersection_area / union_area
    return iou