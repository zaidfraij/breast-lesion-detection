import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    """
    Calculate the IoU (Intersection over Union) between two sets of bounding boxes.
    
    Args:
        a (Tensor): Tensor of shape [N, 4] containing N bounding boxes [x1, y1, x2, y2].
        b (Tensor): Tensor of shape [M, 4] containing M bounding boxes [x1, y1, x2, y2].

    Returns:
        Tensor: IoU values of shape [N, M].
    """
    # Validate inputs
    assert a.ndim == 2 and a.shape[1] == 4, f"Expected 'a' to have shape [N, 4], but got {a.shape}"
    assert b.ndim == 2 and b.shape[1] == 4, f"Expected 'b' to have shape [M, 4], but got {b.shape}"

    # Compute intersection
    iw = torch.min(a[:, None, 2], b[:, 2]) - torch.max(a[:, None, 0], b[:, 0])
    ih = torch.min(a[:, None, 3], b[:, 3]) - torch.max(a[:, None, 1], b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih

    # Compute union
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b - intersection

    union = torch.clamp(union, min=1e-8)  # Avoid division by zero

    # Compute IoU
    IoU = intersection / union
    return IoU


class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


class SequenceFocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0

        # Validate inputs
        assert classifications.ndim == 3, f"Expected 3D classifications, got {classifications.shape}"
        assert regressions.ndim == 3, f"Expected 3D regressions, got {regressions.shape}"
        assert anchors.ndim == 3 and anchors.shape[-1] == 4, f"Expected anchors [B, T, N, 4], got {anchors.shape}"

        # Adjust annotation validation
        if annotations.ndim == 3:
            assert annotations.shape[-1] == 5, f"Expected annotations [B, T, M, 5], got {annotations.shape}"
        else:
            raise ValueError(f"Unexpected annotations shape: {annotations.shape}")

        batch_size, num_anchors, num_classes = classifications.shape
        classification_losses = []
        regression_losses = []
        for b in range(batch_size):
            classification = classifications[b]  # [N, C]
            regression = regressions[b]          # [N, 4]
            anchor = anchors[b]  # [N, 4], remove time dimension
            annotation = annotations[b]  # [M, 5]

            # Ensure anchors have correct shape
            assert anchor.ndim == 2 and anchor.shape[1] == 4, f"Anchor shape mismatch: {anchor.shape}"

            # Filter valid annotations
            annotation = annotation[annotation[:, 4] != -1]  # [M, 5]
            if annotation.shape[0] == 0:  # No valid annotations
                continue

            # Compute IoU
            IoU = calc_iou(anchor, annotation[:, :4])  # Shape: [N, M]
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # Initialize targets
            targets = torch.ones((num_anchors, num_classes), device=classification.device) * -1

            # Ensure background_mask matches the first dimension of targets
            # Create a mask for background anchors
            background_mask = IoU_max < 0.4

            # Expand the mask to match the targets shape
            background_mask = background_mask.unsqueeze(-1).expand(-1, targets.shape[1])

            # Use the mask to set background targets
            targets[background_mask] = 0

            # Set positive targets
            positive_indices = IoU_max >= 0.5
            num_positive_anchors = positive_indices.sum()

            if num_positive_anchors > 0:
                assigned_annotations = annotation[IoU_argmax[positive_indices]]
                targets[positive_indices, :] = 0
                targets[positive_indices, assigned_annotations[:, 4].long()] = 1

                classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
                alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
                focal_weight = torch.where(targets == 1, 1 - classification, classification)
                focal_weight = alpha_factor * (focal_weight ** gamma)
                bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                cls_loss = torch.where(targets != -1, cls_loss, torch.zeros_like(cls_loss))
                if cls_loss.numel() > 0:
                    classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
                else:
                    classification_losses.append(torch.tensor(0.0, device=classification.device))
            else:
                # No positive anchors, append 0 loss
                classification_losses.append(torch.tensor(0.0, device=classification.device))

            # Regression loss
            if num_positive_anchors > 0:
                # Assigned annotations are already matched to positive anchors
                assigned_annotations = annotation[IoU_argmax[positive_indices]]
                anchor = anchor[positive_indices]
                regression = regression[positive_indices]

                anchor_widths = anchor[:, 2] - anchor[:, 0]
                anchor_heights = anchor[:, 3] - anchor[:, 1]
                anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
                anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
                targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
                targets_dw = torch.log(gt_widths / anchor_widths)
                targets_dh = torch.log(gt_heights / anchor_heights)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
                targets = targets / torch.tensor([0.1, 0.1, 0.2, 0.2], device=targets.device)

                regression_diff = torch.abs(targets - regression)
                reg_loss = torch.where(
                    regression_diff < 1.0 / 9.0,
                    0.5 * 9.0 * regression_diff ** 2,
                    regression_diff - 0.5 / 9.0
                )
                if reg_loss.numel() > 0:
                    regression_losses.append(reg_loss.mean())
                else:
                    regression_losses.append(torch.tensor(0.0, device=classification.device))
            else:
                regression_losses.append(torch.tensor(0.0, device=classification.device))


        # Ensure the lists are not empty before stacking
        if not classification_losses:
            print("No classification losses")
            classification_losses.append(torch.tensor(0.0, device=classification.device))
        if not regression_losses:
            print("No regression losses")
            regression_losses.append(torch.tensor(0.0, device=classification.device))

        return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()
