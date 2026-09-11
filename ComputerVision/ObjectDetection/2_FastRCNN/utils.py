import os

import cv2
import numpy as np


def compute_iou(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    boxes_a = boxes_a.astype(np.float32)
    boxes_b = boxes_b.astype(np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * \
        np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * \
        np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0.0, intersection / union, 0.0).astype(np.float32)


def clip_box(box, height, width):
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = np.clip(x1, 0.0, max(width - 1, 0))
    y1 = np.clip(y1, 0.0, max(height - 1, 0))
    x2 = np.clip(x2, x1 + 1.0, max(width, 1))
    y2 = np.clip(y2, y1 + 1.0, max(height, 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def deduplicate_boxes(boxes):
    if len(boxes) == 0:
        return boxes.astype(np.float32)
    boxes = np.round(boxes).astype(np.int32)
    _, unique_indices = np.unique(boxes, axis=0, return_index=True)
    return boxes[np.sort(unique_indices)].astype(np.float32)


def generate_grid_proposals(height, width, max_proposals):
    """Generate a deterministic, scale/aspect-balanced fallback proposal set.

    Selective search is optional in this project.  When it is unavailable, do
    not let the dense smallest windows consume the proposal budget before the
    larger scale/aspect families are visited.
    """
    if max_proposals <= 0 or height < 1 or width < 1:
        return np.empty((0, 4), dtype=np.float32)

    scales = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    aspect_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
    families = []
    for scale in scales:
        base = max(16, int(min(height, width) * scale))
        stride = max(8, base // 3)
        for ratio in aspect_ratios:
            box_w = int(np.sqrt(base * base * ratio))
            box_h = int(max(16, base * base / max(box_w, 1)))
            box_w = min(box_w, width)
            box_h = min(box_h, height)
            if box_w < 16 or box_h < 16:
                continue
            family = []
            for y in range(0, max(height - box_h + 1, 1), stride):
                for x in range(0, max(width - box_w + 1, 1), stride):
                    family.append([x, y, x + box_w, y + box_h])
            if family:
                families.append(deduplicate_boxes(
                    np.asarray(family, dtype=np.float32)))

    # Reserve one slot for the whole image, then give every valid family the
    # same quota (with deterministic remainder assignment).
    full_image = np.array([[0, 0, width, height]], dtype=np.float32)
    budget = max(max_proposals - 1, 0)
    quotas = [budget // len(families) if families else 0] * len(families)
    for index in range(budget % len(families) if families else 0):
        quotas[index] += 1

    selected = []
    leftovers = []
    for family, quota in zip(families, quotas):
        take = min(quota, len(family))
        if take:
            # Evenly spaced indices retain spatial coverage within a family.
            indices = np.linspace(0, len(family) - 1, num=take, dtype=np.int32)
            selected.extend(family[indices])
            selected_indices = set(indices.tolist())
            leftovers.extend(box for index, box in enumerate(
                family) if index not in selected_indices)
        else:
            leftovers.extend(family)

    # Tiny images can leave a family short of its quota.  Fill only those
    # unused slots, keeping the family-balanced allocation whenever possible.
    if len(selected) < budget:
        selected.extend(leftovers[: budget - len(selected)])
    proposals = deduplicate_boxes(
        np.vstack([np.asarray(selected, dtype=np.float32), full_image]))
    return proposals[:max_proposals]


def generate_region_proposals(image_bgr, max_proposals=2000, use_selective_search=True):
    height, width = image_bgr.shape[:2]
    if use_selective_search and hasattr(cv2, "ximgproc"):
        segmentation = getattr(cv2.ximgproc, "segmentation", None)
        if segmentation is not None and hasattr(segmentation, "createSelectiveSearchSegmentation"):
            selective_search = segmentation.createSelectiveSearchSegmentation()
            selective_search.setBaseImage(image_bgr)
            selective_search.switchToSelectiveSearchFast()
            rects = selective_search.process()
            proposals = []
            for x, y, w, h in rects[: max_proposals * 2]:
                if w < 16 or h < 16:
                    continue
                proposals.append([x, y, x + w, y + h])
            proposals = deduplicate_boxes(
                np.array(proposals, dtype=np.float32))
            if len(proposals) > 0:
                return proposals[:max_proposals]
    return generate_grid_proposals(height=height, width=width, max_proposals=max_proposals)


def encode_box(proposal_box, gt_box):
    proposal_box = proposal_box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)
    px1, py1, px2, py2 = proposal_box
    gx1, gy1, gx2, gy2 = gt_box
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)
    gw = max(gx2 - gx1, 1.0)
    gh = max(gy2 - gy1, 1.0)
    pcx = px1 + 0.5 * pw
    pcy = py1 + 0.5 * ph
    gcx = gx1 + 0.5 * gw
    gcy = gy1 + 0.5 * gh
    tx = (gcx - pcx) / pw
    ty = (gcy - pcy) / ph
    tw = np.log(gw / pw)
    th = np.log(gh / ph)
    return np.array([tx, ty, tw, th], dtype=np.float32)


def decode_box(proposal_box, deltas):
    proposal_box = proposal_box.astype(np.float32)
    deltas = deltas.astype(np.float32)
    px1, py1, px2, py2 = proposal_box
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)
    pcx = px1 + 0.5 * pw
    pcy = py1 + 0.5 * ph
    tx, ty, tw, th = deltas
    gcx = pcx + tx * pw
    gcy = pcy + ty * ph
    gw = pw * np.exp(np.clip(tw, -4.0, 4.0))
    gh = ph * np.exp(np.clip(th, -4.0, 4.0))
    x1 = gcx - 0.5 * gw
    y1 = gcy - 0.5 * gh
    x2 = gcx + 0.5 * gw
    y2 = gcy + 0.5 * gh
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def nms(boxes, scores, iou_threshold=0.3, max_keep=100):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_keep:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = compute_iou(boxes[idx: idx + 1], boxes[remaining]).reshape(-1)
        order = remaining[ious < iou_threshold]
    return np.array(keep, dtype=np.int32)


def batched(array, batch_size):
    for start in range(0, len(array), batch_size):
        yield array[start: start + batch_size]


def draw_detections(image_bgr, detections, class_names, max_draw=20):
    canvas = image_bgr.copy()
    for detection in detections[:max_draw]:
        x1, y1, x2, y2 = detection["box"].astype(np.int32)
        score = float(detection["score"])
        class_id = int(detection["class_id"])
        label = class_names[class_id]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, f"{label}: {score:.2f}", (x1, max(
            15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    return canvas


def draw_ground_truth(image_bgr, boxes, labels, class_names, max_draw=20):
    """Overlay ground-truth boxes in red for training inspection images."""
    canvas = image_bgr.copy()
    for box, class_id in zip(boxes[:max_draw], labels[:max_draw]):
        x1, y1, x2, y2 = np.asarray(box, dtype=np.int32)
        class_id = int(class_id)
        label = class_names[class_id] if 0 <= class_id < len(
            class_names) else str(class_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"GT {label}", (x1, max(
            15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def make_output_path(output_dir, image_path):
    basename = os.path.basename(image_path)
    stem, ext = os.path.splitext(basename)
    if not ext:
        ext = ".jpg"
    return os.path.join(output_dir, f"{stem}_pred{ext}")
