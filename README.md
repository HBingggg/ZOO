# ZOO

**ZOO: A Training-Free Framework for Open-Vocabulary Animal Detection, Segmentation, and Tracking**

ZOO (Zero-shot Organism Observation) is a training-free workflow for animal detection, instance segmentation, and multi-object tracking. It combines Grounding DINO with SAM 2.1 and provides two entry points:

- `zoo_image.py`: detection and instance segmentation for images.
- `zoo_video.py`: detection, segmentation, and trajectory tracking for videos.

No additional training or fine-tuning is required for a target species or recording environment.

## Tested environment

The experiments reported in the manuscript used:

- Ubuntu 22.04
- Python 3.9.18
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA 11.8
- NVIDIA RTX 4090D
- NumPy 1.26.0
- OpenCV 4.8.0
- timm 0.9.7
- transformers 4.33.2

An NVIDIA GPU is strongly recommended. Runtime depends on image/video resolution, scene complexity, and the number of detected animals.

## Installation

```bash
conda create -n zoo python=3.9.18
conda activate zoo

# Grounding DINO compiles a CUDA extension. If nvcc is not on PATH,
# set CUDA_HOME and PATH to your local CUDA 11.8 installation first.

# Install PyTorch 2.0.1 and torchvision 0.15.2 with CUDA 11.8,
# then install the remaining dependencies.
pip install -r grounding_dino/requirements.txt
pip install scipy filterpy pyyaml hydra-core omegaconf iopath tqdm pillow
pip install -e ./grounding_dino
```

## Model checkpoints

Download the pretrained checkpoints and place them at the paths used by `config.yaml`:

```text
gdino_checkpoints/groundingdino_swint_ogc.pth
checkpoints/sam2.1_hiera_large.pt
```

The default model configurations are:

```text
grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
configs/sam2.1/sam2.1_hiera_l.yaml
```

The SAM 2 configuration is resolved from the bundled `sam2` package (the corresponding repository file is `sam2/configs/sam2.1/sam2.1_hiera_l.yaml`).

If different files are used, update `grounding_dino.config_path`, `grounding_dino.checkpoint`, `sam2.config`, and `sam2.checkpoint` in `config.yaml`.

## Quick start

Run all commands from the repository root.

### ZOO-Image

```bash
python zoo_image.py \
  --image_path /path/to/image.jpg \
  --text_prompt "zebra" \
  --config_path config.yaml \
  --image_save true \
  --output outputs/image
```

The image pipeline saves a JSON result file, instance masks (`npz` or `png`), and an optional annotated image.

### ZOO-Video

```bash
python zoo_video.py \
  --video_path /path/to/video.mp4 \
  --text_prompt "zebra" \
  --config_path config.yaml \
  --save_video true \
  --output_path outputs/video
```

The video pipeline saves trajectory results in JSON format, frame-associated instance masks, and an optional annotated video.

## Main command-line arguments

### `zoo_image.py`

| Argument | Description | Default |
| --- | --- | --- |
| `--image_path` | Input image path | required |
| `--text_prompt` | Animal name or open-vocabulary prompt | value in `config.yaml` |
| `--config_path` | YAML configuration file | optional |
| `--box_threshold` | Grounding DINO box threshold | `0.25` |
| `--text_threshold` | Grounding DINO text threshold | `0.35` |
| `--image_save` | Save an annotated image | `false` |
| `--output` | Output directory | `outputs` |

### `zoo_video.py`

| Argument | Description | Default |
| --- | --- | --- |
| `--video_path` | Input video path | required |
| `--text_prompt` | Animal name or open-vocabulary prompt | value in `config.yaml` |
| `--config_path` | YAML configuration file | required |
| `--save_video` | Save an annotated video | `false` |
| `--output_path` | Output directory | `outputs` |

## Main configuration parameters

The default settings are defined in the root `config.yaml`.

| Parameter | Default | Description |
| --- | ---: | --- |
| `grounding_dino.text_prompt` | `animal` | Default text prompt |
| `grounding_dino.box_threshold` | `0.25` | Detection confidence threshold |
| `grounding_dino.text_threshold` | `0.35` | Text-matching threshold |
| `image_filter.mask_iou_threshold` | `0.8` | Suppresses strongly overlapping instance masks |
| `video.mask_filter_interval` | `5` | Interval between mask-overlap filtering frames |
| `tracker.det_thresh` | `0.4` | Tracker detection threshold |
| `tracker.iou_threshold` | `0.3` | IoU threshold for track association |
| `tracker.max_age` | `30` | Maximum unmatched track age |
| `tracker.min_hits` | `3` | Minimum hits used to confirm a track |
| `tracker.occlusion_penalty` | `0.5` | Penalty applied during occlusion-aware association |
| `optical_flow.process_size` | `[480, 480]` | Resolution used for optical-flow processing |
| `optical_flow.overlap_threshold` | `0.3` | Overlap threshold used during mask propagation |

Masks are saved by default. Set `image.save_mask` or `video.save_mask` to `false` to disable mask export, and set `mask_format` to either `npz` or `png`.

Optional laboratory-scene priors and an external identity validator are available through the `tracker` and `id_validator` sections of `config.yaml`; both are disabled by default.

## Notes

- Text prompts can contain a species name, a broader taxonomic label, or multiple labels, depending on the analysis goal.
- Thresholds may need adjustment for small, occluded, infrared, or densely grouped animals.
- Model checkpoints, datasets, and generated outputs are excluded from version control by `.gitignore`.
