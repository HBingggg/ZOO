# zoo_video.py
import argparse
import os
import torch
from core.video_filter import VideoProcessor
from core.utils import load_config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZOOE Video Tracking Pipeline")

    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--text_prompt", type=str, default=None, help="Animal category for detection")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output directory")
    parser.add_argument("--save_video", type=str2bool, default=False, help="Whether to generate annotated video")

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: Config file {args.config_path} not found.")
        exit(1)

    config = load_config(args.config_path)

    if args.text_prompt:
        config.setdefault('grounding_dino', {})['text_prompt'] = args.text_prompt

    text_prompt = config.get('grounding_dino', {}).get('text_prompt')
    if not text_prompt:
        raise ValueError("A text prompt must be provided via --text_prompt or grounding_dino.text_prompt in config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = VideoProcessor(config, device=device)
    processor.process_video(
        video_path=args.video_path,
        text_prompt=text_prompt,
        output_path=args.output_path,
        save_video=args.save_video
    )
