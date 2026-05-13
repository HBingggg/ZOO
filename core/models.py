# core/models.py
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model


class ZOOEModelManager:
    """
    Centralized model loader for GroundingDINO and SAM2.

    The default checkpoint/config paths are kept backward compatible with the
    original scripts, but they can now be overridden from a config dictionary:

    sam2:
      checkpoint: ./checkpoints/sam2.1_hiera_large.pt
      config: configs/sam2.1/sam2.1_hiera_l.yaml
    grounding_dino:
      config_path: grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
      checkpoint: gdino_checkpoints/groundingdino_swint_ogc.pth
    """

    def __init__(self,
                 sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
                 sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 gdino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 gdino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
                 device="cuda",
                 config=None):
        config = config or {}
        sam2_cfg = config.get("sam2", {}) or {}
        gdino_cfg = config.get("grounding_dino", {}) or {}

        sam2_checkpoint = sam2_cfg.get("checkpoint", sam2_checkpoint)
        sam2_config = sam2_cfg.get("config", sam2_config)
        gdino_config = gdino_cfg.get("config_path", gdino_config)
        gdino_checkpoint = gdino_cfg.get("checkpoint", gdino_checkpoint)

        self.device = device
        print(f"[ModelManager] Loading models on {device}...")

        print("  - Loading GroundingDINO...")
        self.gdino = load_model(gdino_config, gdino_checkpoint, device=device)

        print("  - Loading SAM2 Image Predictor...")
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        print("[ModelManager] Ready.")

    def get_gdino(self):
        return self.gdino

    def get_sam2_predictor(self):
        return self.sam2_predictor
