from pycocotools.cocoeval import COCOeval
import json
import torch
import os


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
