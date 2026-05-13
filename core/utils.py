# core/utils.py
import yaml
import cv2
import numpy as np
import importlib.util
import sys
import os


def load_config(config_path):
    """Load YAML config. Return an empty dict if config_path is None."""
    if config_path is None:
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_cfg(config, path, default=None):
    """Safely read nested config keys using dot-separated paths."""
    cur = config or {}
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_external_function(file_path, function_name):
    """Dynamically load a function from an external Python file."""
    if not os.path.exists(file_path):
        print(f"Warning: Validator file {file_path} not found.")
        return None

    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, function_name):
        return getattr(module, function_name)
    print(f"Warning: Function {function_name} not found in {file_path}.")
    return None


def process_lab_scene(frame, background_img):
    """Simple fixed-background subtraction for laboratory scenes."""
    if background_img is None:
        return frame

    if frame.shape != background_img.shape:
        background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))

    diff = cv2.absdiff(frame, background_img)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)
    return foreground


def shift_mask(mask, prev_box, curr_box):
    """Translate a mask according to the displacement between two boxes."""
    if mask is None:
        return None

    h, w = mask.shape[:2]

    prev_center_x = (prev_box[0] + prev_box[2]) / 2
    prev_center_y = (prev_box[1] + prev_box[3]) / 2
    curr_center_x = (curr_box[0] + curr_box[2]) / 2
    curr_center_y = (curr_box[1] + curr_box[3]) / 2

    dx = int(curr_center_x - prev_center_x)
    dy = int(curr_center_y - prev_center_y)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted_mask
