# zoo_image.py
import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_image, predict as gdino_predict

from core.models import ZOOEModelManager
from core.image_filter import image_filter
from core.utils import load_config, get_cfg


class ZOOEImageDetector:
    def __init__(self, device="cuda", config=None):
        self.config = config or {}
        self.models = ZOOEModelManager(device=device, config=self.config)
        self.device = device

    def predict(self, image_path, text_prompt=None, box_threshold=None, text_threshold=None, image_save=False,
                output_dir="outputs"):
        if not os.path.exists(image_path):
            print(f"Error: Image path '{image_path}' not found.")
            return

        gdino_cfg = self.config.get('grounding_dino', {}) or {}
        img_filter_cfg = self.config.get('image_filter', {}) or {}
        image_cfg = self.config.get('image', {}) or {}

        text_prompt = text_prompt or gdino_cfg.get('text_prompt', 'animal')
        box_threshold = box_threshold if box_threshold is not None else gdino_cfg.get('box_threshold', 0.25)
        text_threshold = text_threshold if text_threshold is not None else gdino_cfg.get('text_threshold', 0.35)
        mask_iou_threshold = img_filter_cfg.get('mask_iou_threshold', 0.8)
        save_masks = bool(image_cfg.get('save_mask', True))
        mask_format = str(image_cfg.get('mask_format', 'npz')).lower()
        if mask_format not in {"npz", "png"}:
            raise ValueError("image.mask_format must be 'npz' or 'png'")

        image_source, image = load_image(image_path)

        print(f"Running GroundingDINO for '{text_prompt}'...")
        boxes, confidences, labels = gdino_predict(
            model=self.models.get_gdino(),
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        if len(boxes) == 0:
            print("No objects detected.")
            return

        h, w, _ = image_source.shape
        boxes_scaled = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
        boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        scores = confidences.detach().cpu().numpy()

        print("Filtering masks with SAM2...")
        sam2_predictor = self.models.get_sam2_predictor()
        filtered_objects = image_filter(
            image=image_source,
            boxes=boxes_xyxy,
            scores=scores,
            labels=labels,
            sam2_predictor=sam2_predictor,
            iou_threshold=mask_iou_threshold
        )

        if len(filtered_objects) == 0:
            print("No objects kept after filtering.")
            return

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}_results.json")

        mask_refs = {}
        if save_masks:
            mask_dir = os.path.join(output_dir, f"{base_name}_masks")
            os.makedirs(mask_dir, exist_ok=True)
            if mask_format == "png":
                for i, obj in enumerate(filtered_objects):
                    mask_name = f"mask_{i:03d}.png"
                    mask_path = os.path.join(mask_dir, mask_name)
                    cv2.imwrite(mask_path, (obj['mask'] > 0).astype(np.uint8) * 255)
                    mask_refs[i] = os.path.relpath(mask_path, output_dir)
            else:
                masks = np.stack([(obj['mask'] > 0).astype(np.uint8) for obj in filtered_objects], axis=0)
                npz_path = os.path.join(mask_dir, f"{base_name}_masks.npz")
                np.savez_compressed(npz_path, masks=masks)
                for i in range(len(filtered_objects)):
                    mask_refs[i] = {"path": os.path.relpath(npz_path, output_dir), "index": i}

        results_data = {
            "image_path": image_path,
            "text_prompt": text_prompt,
            "box_threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "mask_iou_threshold": float(mask_iou_threshold),
            "count": len(filtered_objects),
            "objects": []
        }

        for i, obj in enumerate(filtered_objects):
            results_data["objects"].append({
                "id": i,
                "box": obj["box"],
                "input_box": obj["input_box"],
                "score": obj["score"],
                "label": obj["label"],
                "sam_score": obj["sam_score"],
                "source_index": obj["source_index"],
                "mask_ref": mask_refs.get(i)
            })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        print(f"Saved detection results to {json_path}")

        if image_save:
            print("Saving annotated image...")
            boxes_np = np.array([obj['box'] for obj in filtered_objects], dtype=np.float32)
            masks_np = np.array([obj['mask'].astype(bool) for obj in filtered_objects])
            tracker_ids = np.arange(len(filtered_objects))

            detections = sv.Detections(
                xyxy=boxes_np,
                mask=masks_np,
                tracker_id=tracker_ids
            )

            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()

            annotated_frame = image_source.copy()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            labels_vis = [f"ID:{i}" for i in range(len(filtered_objects))]
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_vis)

            save_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, annotated_frame_bgr)
            print(f"Saved annotated image to {save_path}")

        return results_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZOOE image pipeline")

    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--text_prompt", type=str, default=None, help="animal species or prompt")
    parser.add_argument("--config_path", type=str, default=None, help="Optional path to config.yaml")
    parser.add_argument("--box_threshold", type=float, default=None, help="GroundingDINO box threshold")
    parser.add_argument("--text_threshold", type=float, default=None, help="GroundingDINO text threshold")
    parser.add_argument("--image_save", type=str2bool, default=False,
                        help="Whether to save annotated image (true/false)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    config = load_config(args.config_path) if args.config_path else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = ZOOEImageDetector(device=device, config=config)
    detector.predict(
        image_path=args.image_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        image_save=args.image_save,
        output_dir=args.output
    )
