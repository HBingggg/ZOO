# core/image_filter.py
import numpy as np
import torch


def compute_mask_iou(mask1, mask2):
    """Compute pixel-level IoU between two binary masks."""
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def mask_to_box(mask):
    """
    Convert a binary mask to a continuous xyxy box.

    Returns:
        list | None: [x1, y1, x2, y2), where x2/y2 are exclusive bounds.
        This coordinate convention is consistent with common detection code
        that computes width = x2 - x1 and height = y2 - y1.
    """
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return [int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)]


def _to_numpy_boxes(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    return np.asarray(boxes, dtype=np.float32)


def _normalize_optional_sequence(values, n, default=None):
    if values is None:
        return [default] * n
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = list(values)
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)}")
    return values


def image_filter(image, boxes, sam2_predictor, iou_threshold=0.8, scores=None, labels=None):
    """
    Generate SAM2 masks from detection boxes and suppress duplicate masks.

    Important design choice:
        GroundingDINO confidence is recorded but is NOT used to rank candidates
        for suppression, because open-vocabulary confidence is not always a
        stable proxy for instance-level localization quality. The filter keeps
        the detector/SAM output order and performs greedy suppression based on
        mask topology (mask IoU).

    Args:
        image (np.ndarray): RGB image of shape (H, W, 3).
        boxes (np.ndarray | torch.Tensor): xyxy boxes of shape (N, 4).
        sam2_predictor: initialized SAM2ImagePredictor.
        iou_threshold (float): mask IoU threshold above which a candidate is
            treated as duplicate/nested and suppressed.
        scores (Sequence | None): optional detection scores to preserve in the
            returned records; not used for ordering.
        labels (Sequence | None): optional text labels/phrases to preserve.

    Returns:
        list[dict]: each dict contains mask, box, input_box, score, label,
        source_index and sam_score.
    """
    if len(boxes) == 0:
        print("[Image Filter] No boxes provided.")
        return []

    boxes = _to_numpy_boxes(boxes)
    n = len(boxes)
    scores = _normalize_optional_sequence(scores, n, default=None)
    labels = _normalize_optional_sequence(labels, n, default=None)

    sam2_predictor.set_image(image)

    raw_masks, sam_scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False
    )

    if raw_masks.ndim == 4:
        raw_masks = raw_masks.squeeze(1)

    if isinstance(sam_scores, torch.Tensor):
        sam_scores = sam_scores.detach().cpu().numpy()
    sam_scores = np.asarray(sam_scores).reshape(-1) if sam_scores is not None else np.array([np.nan] * n)

    kept = []
    kept_masks = []

    for i, current_mask in enumerate(raw_masks):
        binary_mask = (current_mask > 0.0).astype(np.uint8)

        is_duplicate = False
        for kept_mask in kept_masks:
            iou = compute_mask_iou(binary_mask, kept_mask)
            if iou > iou_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        mask_box = mask_to_box(binary_mask)
        if mask_box is None:
            continue

        kept_masks.append(binary_mask)
        kept.append({
            "mask": binary_mask,
            "box": mask_box,
            "input_box": boxes[i].astype(float).tolist(),
            "score": None if scores[i] is None else float(scores[i]),
            "label": None if labels[i] is None else str(labels[i]),
            "source_index": int(i),
            "sam_score": None if i >= len(sam_scores) or np.isnan(sam_scores[i]) else float(sam_scores[i])
        })

    print(f"[Image Filter] Processed {len(boxes)} boxes, kept {len(kept)} unique masks.")
    return kept
