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
                            'category_id' : dataset.label_to_coco_label(label),
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
