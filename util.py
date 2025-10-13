import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

import torch
import torchvision.utils as vutils
import torchvision.transforms as T


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed:", seed)


def init_experiment(args, prefix):
    if args.seed is None:
        args.seed = random.randint(0, 10000)

    set_seed(args.seed)

    if not args.name:
        args.name = datetime.now().strftime('%Y%m%d%H%M%S%f')

    out_dir = Path('output') / args.dataset / prefix / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'args.json'
    with json_path.open('w') as f:
        json.dump(vars(args), f, indent=2)

    return out_dir


def save_checkpoint(state, is_best, out_dir):
    out_path = Path(out_dir) / 'checkpoint.pth.tar'
    torch.save(state, out_path)

    if is_best:
        best_path = Path(out_dir) / 'model_best.pth.tar'
        shutil.copyfile(out_path, best_path)


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def convert_ltrb_to_xywh(ltrb):
    """Convert from corners to center format"""
    x1, y1, x2, y2 = ltrb
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [xc, yc, w, h]


def check_overlap(box1, box2, margin=0.01):
    """Check if two boxes (in xywh format) overlap"""
    x1_1, y1_1, x2_1, y2_1 = convert_xywh_to_ltrb(box1)
    x1_2, y1_2, x2_2, y2_2 = convert_xywh_to_ltrb(box2)
    if x2_1 + margin < x1_2 or x2_2 + margin < x1_1 or \
       y2_1 + margin < y1_2 or y2_2 + margin < y1_1:
        return False
    return True


def fix_overlaps(boxes, max_attempts=200, move_step=0.005, min_dist=0.015):
    """
    Fix overlapping boxes by iteratively moving them apart.
    Only used during inference (generate.py), not during training.
    
    Args:
        boxes: numpy array of shape (N, 4) in xywh format
        max_attempts: maximum iterations to fix overlaps
        move_step: distance to move boxes in each iteration
        min_dist: minimum distance between box edges
        
    Returns:
        fixed_boxes: list of boxes in xywh format
        success: whether all overlaps were fixed
    """
    boxes = np.array(boxes, dtype=float).copy()
    n_boxes = len(boxes)
    fixed_all = False
    
    for attempt in range(max_attempts):
        overlap_found_in_iteration = False
        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                if check_overlap(boxes[i], boxes[j], margin=min_dist):
                    overlap_found_in_iteration = True
                    # Calculate direction vector between centers
                    c_i_x, c_i_y = boxes[i][0], boxes[i][1]
                    c_j_x, c_j_y = boxes[j][0], boxes[j][1]
                    vec_x, vec_y = c_i_x - c_j_x, c_i_y - c_j_y
                    dist = np.sqrt(vec_x**2 + vec_y**2)
                    
                    # Avoid division by zero
                    if dist < 1e-6:
                        vec_x, dist = 0.01, 0.01
                    
                    unit_vec_x, unit_vec_y = vec_x / dist, vec_y / dist
                    
                    # Move boxes apart along the direction vector
                    boxes[i][0] += unit_vec_x * move_step
                    boxes[i][1] += unit_vec_y * move_step
                    boxes[j][0] -= unit_vec_x * move_step
                    boxes[j][1] -= unit_vec_y * move_step
                    
                    # Keep boxes within valid bounds [0, 1]
                    boxes[i][0] = np.clip(boxes[i][0], boxes[i][2]/2, 1 - boxes[i][2]/2)
                    boxes[i][1] = np.clip(boxes[i][1], boxes[i][3]/2, 1 - boxes[i][3]/2)
                    boxes[j][0] = np.clip(boxes[j][0], boxes[j][2]/2, 1 - boxes[j][2]/2)
                    boxes[j][1] = np.clip(boxes[j][1], boxes[j][3]/2, 1 - boxes[j][3]/2)
        
        if not overlap_found_in_iteration:
            fixed_all = True
            break
    
    return boxes.tolist(), fixed_all


def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color,
                       fill=c_fill)
    return img


def save_image(batch_boxes, batch_labels, batch_mask,
               dataset_colors, out_path, canvas_size=(60, 40),
               nrow=None):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes = batch_boxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        img = convert_layout_to_image(boxes, labels,
                                      dataset_colors,
                                      canvas_size)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    vutils.save_image(image, out_path, normalize=False, nrow=nrow)
