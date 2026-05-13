# core/video_filter.py
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import predict as gdino_predict

from core.models import ZOOEModelManager
from core.Tracker import HybridOCSORT
from core.flow import OpticalFlowTracker
from core.image_filter import image_filter
from core.utils import process_lab_scene, shift_mask, load_external_function, get_cfg


def _as_float_list(values):
    return [float(v) for v in values]


def _ensure_mask_dir(output_path, video_name):
    mask_dir = os.path.join(output_path, f"{video_name}_masks")
    os.makedirs(mask_dir, exist_ok=True)
    return mask_dir


def _save_frame_masks(mask_dir, frame_idx, masks_data, mask_format="npz"):
    """Save all masks for one frame and return object-level mask references."""
    refs = {}
    if not masks_data:
        return refs

    if mask_format == "png":
        frame_dir = os.path.join(mask_dir, f"frame_{frame_idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        for obj in masks_data:
            mask = obj.get("mask")
            if mask is None:
                continue
            mask_name = f"track_{obj['track_id']}.png"
            mask_path = os.path.join(frame_dir, mask_name)
            cv2.imwrite(mask_path, (mask > 0).astype(np.uint8) * 255)
            refs[obj["track_id"]] = os.path.relpath(mask_path, os.path.dirname(mask_dir))
        return refs

    # default: one compressed NPZ per frame
    masks = []
    track_ids = []
    for obj in masks_data:
        mask = obj.get("mask")
        if mask is None:
            continue
        masks.append((mask > 0).astype(np.uint8))
        track_ids.append(int(obj["track_id"]))

    if not masks:
        return refs

    npz_name = f"frame_{frame_idx:06d}.npz"
    npz_path = os.path.join(mask_dir, npz_name)
    np.savez_compressed(npz_path, masks=np.stack(masks, axis=0), track_ids=np.array(track_ids, dtype=np.int64))
    for idx, track_id in enumerate(track_ids):
        refs[track_id] = {
            "path": os.path.relpath(npz_path, os.path.dirname(mask_dir)),
            "index": int(idx)
        }
    return refs


class VideoProcessor:
    def __init__(self, config, device="cuda"):
        self.config = config or {}
        self.device = device

        self.model_manager = ZOOEModelManager(device=device, config=self.config)
        self.gdino = self.model_manager.get_gdino()
        self.sam2_predictor = self.model_manager.get_sam2_predictor()

        validator_func = None
        id_cfg = self.config.get('id_validator', {}) or {}
        enable_validator = bool(id_cfg.get('if_validator', False))

        if enable_validator:
            v_path = id_cfg.get('id_validator_path', '')
            v_func_name = id_cfg.get('id_validator_function', '') or id_cfg.get('id_validator', 'locate_from_box')
            print(f"Loading validator from {v_path}...")
            validator_func = load_external_function(v_path, v_func_name)
            if validator_func:
                print("Validator loaded successfully.")
            else:
                print("Validator failed to load, disabling validation.")
                enable_validator = False

        t_cfg = self.config.get('tracker', {}) or {}
        total_target_count = t_cfg.get('total_target_count', None)
        if total_target_count in ("", 0, "0") and not t_cfg.get('is_lab_scene', False):
            total_target_count = None

        self.tracker = HybridOCSORT(
            is_lab_scene=t_cfg.get('is_lab_scene', False),
            total_target_count=total_target_count,
            det_thresh=t_cfg.get('det_thresh', 0.4),
            iou_threshold=t_cfg.get('iou_threshold', 0.3),
            max_age=t_cfg.get('max_age', 30),
            min_hits=t_cfg.get('min_hits', 3),
            validator=validator_func,
            enable_validator=enable_validator,
            occlusion_penalty=t_cfg.get('occlusion_penalty', 0.5)
        )

        flow_cfg = self.config.get('optical_flow', {}) or {}
        overlap_thresh = flow_cfg.get('overlap_threshold', 0.3)
        self.flow_tracker = OpticalFlowTracker(overlap_threshold=overlap_thresh)

        self.flow_process_size = flow_cfg.get('process_size', [480, 480])
        if self.flow_process_size:
            self.flow_process_size = tuple(self.flow_process_size)

        video_cfg = self.config.get('video', {}) or {}
        self.mask_filter_interval = int(video_cfg.get('mask_filter_interval', 5))
        if self.mask_filter_interval <= 0:
            raise ValueError("video.mask_filter_interval must be a positive integer")
        self.save_masks = bool(video_cfg.get('save_mask', True))
        self.mask_format = str(video_cfg.get('mask_format', 'npz')).lower()
        if self.mask_format not in {"npz", "png"}:
            raise ValueError("video.mask_format must be 'npz' or 'png'")

        self.image_filter_iou = get_cfg(self.config, 'image_filter.mask_iou_threshold', 0.8)

        self.bg_img = None
        if t_cfg.get('is_lab_scene', False) and t_cfg.get('scene_image', ''):
            if os.path.exists(t_cfg['scene_image']):
                self.bg_img = cv2.imread(t_cfg['scene_image'])
            else:
                print(f"Warning: Scene image {t_cfg['scene_image']} not found.")

    def _run_gdino(self, process_frame, text_prompt):
        import grounding_dino.groundingdino.datasets.transforms as T
        from PIL import Image

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pil = Image.fromarray(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform(image_pil, None)

        gdino_cfg = self.config.get('grounding_dino', {}) or {}
        boxes_filt, conf, labels = gdino_predict(
            model=self.gdino,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=gdino_cfg.get('box_threshold', 0.25),
            text_threshold=gdino_cfg.get('text_threshold', 0.35),
            device=self.device
        )

        h_img, w_img, _ = process_frame.shape
        if len(boxes_filt) == 0:
            return np.empty((0, 4), dtype=np.float32), np.array([], dtype=np.float32), []

        boxes_scaled = boxes_filt * torch.Tensor([w_img, h_img, w_img, h_img]).to(boxes_filt.device)
        boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        scores = conf.detach().cpu().numpy().astype(np.float32)
        return boxes_xyxy.astype(np.float32), scores, labels

    def process_video(self, video_path, text_prompt, output_path, save_video=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_path, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mask_dir = _ensure_mask_dir(output_path, video_name) if self.save_masks else None

        video_writer = None
        if save_video:
            out_vid_path = os.path.join(output_path, f"{video_name}_annotated.mp4")
            video_writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()

        frame_idx = 0
        prev_masks_data = []
        prev_frame = None
        trajectory_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_idx}/{total_frames}...")

            if self.config.get('tracker', {}).get('is_lab_scene', False):
                process_frame = process_lab_scene(frame, self.bg_img)
            else:
                process_frame = frame.copy()

            boxes_xyxy, scores, labels = self._run_gdino(process_frame, text_prompt)

            detections = np.empty((0, 5), dtype=np.float32)
            use_mask_filter = (frame_idx % self.mask_filter_interval == 0)

            if use_mask_filter and len(boxes_xyxy) > 0:
                filtered = image_filter(
                    image=cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB),
                    boxes=boxes_xyxy,
                    scores=scores,
                    labels=labels,
                    sam2_predictor=self.sam2_predictor,
                    iou_threshold=self.image_filter_iou
                )
                if filtered:
                    detections = np.array([[*obj['box'], obj['score'] if obj['score'] is not None else 1.0]
                                           for obj in filtered], dtype=np.float32)
            elif len(boxes_xyxy) > 0:
                detections = np.concatenate([boxes_xyxy, scores[:, None]], axis=1).astype(np.float32)

            flow_results = []
            if frame_idx > 0 and prev_frame is not None:
                input_objs = []
                for item in prev_masks_data:
                    input_objs.append({
                        'id': item['track_id'],
                        'mask': item['mask'],
                        'is_occluded': item['is_occluded']
                    })
                if input_objs:
                    flow_results = self.flow_tracker.predict_next_step(
                        prev_frame, process_frame, input_objs, process_size=self.flow_process_size
                    )

            track_results = self.tracker.update(detections, flow_results, image=frame)

            current_masks_data = []
            if track_results:
                self.sam2_predictor.set_image(cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))

            h_img, w_img, _ = process_frame.shape

            for trk in track_results:
                track_id = trk['track_id']
                current_box = np.array(trk['box'], dtype=np.float32)

                prev_data = next((item for item in prev_masks_data if item['track_id'] == track_id), None)

                final_mask = None
                final_occluded_status = int(trk.get('is_occluded', 0))

                if frame_idx == 0:
                    masks, _, _ = self.sam2_predictor.predict(box=current_box[None, :], multimask_output=False)
                    final_mask = (masks.squeeze() > 0).astype(np.uint8)
                    final_occluded_status = 0
                else:
                    if final_occluded_status == 1:
                        if prev_data and prev_data.get('mask') is not None:
                            shifted = shift_mask(prev_data['mask'], prev_data['box'], current_box)
                            final_mask = shifted
                        else:
                            final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    else:
                        masks, _, _ = self.sam2_predictor.predict(box=current_box[None, :], multimask_output=False)
                        final_mask = (masks.squeeze() > 0).astype(np.uint8)

                current_masks_data.append({
                    'track_id': track_id,
                    'id': trk['id'],
                    'validated_id': trk.get('validated_id'),
                    'mask': final_mask,
                    'is_occluded': final_occluded_status,
                    'flow_status': int(trk.get('flow_status', 0)),
                    'box': current_box,
                    'score': trk.get('score')
                })

            mask_refs = _save_frame_masks(mask_dir, frame_idx, current_masks_data, self.mask_format) if self.save_masks else {}

            frame_results_json = []
            for item in current_masks_data:
                obj_json = {
                    "track_id": int(item['track_id']),
                    "id": item['id'],
                    "validated_id": item.get('validated_id'),
                    "box": _as_float_list(item['box']),
                    "score": item.get('score'),
                    "is_occluded": int(item['is_occluded']),
                    "flow_status": int(item.get('flow_status', 0))
                }
                if self.save_masks:
                    obj_json["mask_ref"] = mask_refs.get(item['track_id'])
                frame_results_json.append(obj_json)

            prev_masks_data = current_masks_data
            prev_frame = process_frame.copy()

            trajectory_results.append({
                "frame_id": frame_idx,
                "objects": frame_results_json
            })

            if save_video and video_writer is not None:
                if current_masks_data:
                    xyxy = np.array([d['box'] for d in current_masks_data], dtype=np.float32)
                    masks_np = np.array([
                        d['mask'].astype(bool) if d['mask'] is not None else np.zeros((h_img, w_img), bool)
                        for d in current_masks_data
                    ])
                    ids_np = np.array([int(d['track_id']) for d in current_masks_data])

                    detections_sv = sv.Detections(
                        xyxy=xyxy,
                        mask=masks_np,
                        class_id=ids_np,
                        tracker_id=ids_np
                    )

                    annotated_frame = frame.copy()
                    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections_sv)
                    labels_vis = [f"ID: {d['id']}" for d in current_masks_data]
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections_sv,
                                                             labels=labels_vis)
                else:
                    annotated_frame = frame.copy()

                video_writer.write(annotated_frame)

            frame_idx += 1

        cap.release()
        if video_writer:
            video_writer.release()

        final_trajectory = self.tracker.post_process_trajectories(trajectory_results)

        json_out_path = os.path.join(output_path, f"{video_name}_trajectory.json")
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(final_trajectory, f, indent=4, ensure_ascii=False)
        print(f"Done. Corrected results saved to {json_out_path}")
