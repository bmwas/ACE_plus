#!/usr/bin/env python3
"""
run_inference.py - Demo script for ACE_plus LoRA model inference (converted from bash)

USAGE INSTRUCTIONS
------------------
1. Install dependencies:
   cd ACE_plus
   pip install -r repo_requirements.txt

2. Set your WandB API key in a `.env` file:
   echo "WANDB_API_KEY=your_key_here" >> .env

3. Run the inference script:
   python3.10 run_inference.py \
     --task_type subject \
     --instruction "A beautiful landscape with mountains, clear blue sky, and a lake" \
     --output_h 512 \
     --output_w 512 \
     --seed 42 \
     --input_reference_image ./assets/samples/control/resuzed_balnk.webp \
     --output_dir ./examples/output_images

Options:
  --task_type              portrait, subject, local_editing (default: subject)
  --instruction            prompt text (default shown above)
  --output_h, --output_w   height/width in pixels
  --seed                   random seed for reproducibility
  --input_reference_image  path to reference image
  --output_dir             directory to save outputs

Console logs (INFO/DEBUG/ERROR) will display runtime details. All metrics and generated images are pushed to the WandB project 'ace_plus_inference'.
"""
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize wandb for logging metrics and images
import wandb

# Logging functions with colored output

def log_info(msg):
    print(f"\033[0;32m[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m: {msg}")

def log_warn(msg):
    print(f"\033[0;33m[WARNING] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m: {msg}")

def log_error(msg):
    print(f"\033[0;31m[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m: {msg}")

def log_debug(msg):
    print(f"\033[0;36m[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m: {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="ACE_plus LoRA Model Inference")
    parser.add_argument('--task_type', choices=['portrait', 'subject', 'local_editing'], default='subject',
                        help='Task type: portrait, subject, local_editing')
    parser.add_argument('--instruction', default='A beautiful landscape with mountains, clear blue sky, and a lake',
                        help='Text prompt for generation')
    parser.add_argument('--output_h', type=int, default=512, help='Output height')
    parser.add_argument('--output_w', type=int, default=512, help='Output width')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input_reference_image', default='./assets/samples/control/resuzed_balnk.webp',
                        help='Path to reference image')
    parser.add_argument('--output_dir', default='./examples/output_images',
                        help='Directory to save outputs')
    return parser.parse_args()


def check_dependencies():
    log_info("Checking dependencies...")
    for pkg in ['torch', 'diffusers', 'wandb']:
        try:
            __import__(pkg)
            log_info(f"{pkg} is installed.")
        except ImportError:
            log_error(f"{pkg} is not installed. Please install it via pip.")
            sys.exit(1)
    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log_info(f"GPU available: {name}")
        else:
            log_warn("No GPU available.")
    except Exception as e:
        log_warn(f"Error checking GPU: {e}")


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"generated_{args.task_type}_{ts}.png")

    # Set LoRA model environment variables
    os.environ['PORTRAIT_MODEL_PATH'] = 'ms://iic/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors'
    os.environ['SUBJECT_MODEL_PATH'] = 'ms://iic/ACE_Plus@subject/comfyui_subject_lora16.safetensors'
    os.environ['LOCAL_MODEL_PATH'] = 'ms://iic/ACE_Plus@local_editing/comfyui_local_lora16.safetensors'

    # Log run parameters
    log_info("Starting inference process")
    log_debug(f"Task Type: {args.task_type}")
    log_debug(f"Instruction: {args.instruction}")
    log_debug(f"Output Size: {args.output_h}x{args.output_w}")
    log_debug(f"Seed: {args.seed}")
    log_debug(f"Reference Image: {args.input_reference_image}")
    log_debug(f"Output Dir: {args.output_dir}")
    log_debug(f"Output File: {output_file}")

    # Check dependencies and GPU
    check_dependencies()

    # Initialize wandb
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
    else:
        log_warn("WANDB_API_KEY not found in environment; skipping wandb login.")
    run = wandb.init(project='ace_plus_inference', config={
        'task_type': args.task_type,
        'instruction': args.instruction,
        'output_h': args.output_h,
        'output_w': args.output_w,
        'seed': args.seed,
        'input_reference_image': args.input_reference_image,
        'output_dir': args.output_dir,
    })

    # Build inference command
    cmd = [
        sys.executable, 'infer_lora.py',
        '--instruction', args.instruction,
        '--output_h', str(args.output_h),
        '--output_w', str(args.output_w),
        '--seed', str(args.seed),
        '--task_type', args.task_type,
        '--input_reference_image', args.input_reference_image,
        '--save_path', output_file
    ]
    log_info(f"Command: {' '.join(cmd)}")

    # Execute inference
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if proc.returncode == 0:
        log_info(f"Inference completed successfully in {elapsed:.2f}s")
        if os.path.isfile(output_file):
            size = os.path.getsize(output_file)
            log_debug(f"Generated image size: {size} bytes")
            log_info(f"Image generated: {output_file}")
            # Log metrics and image to wandb
            wandb.log({
                'elapsed_time': elapsed,
                'generated_image': wandb.Image(output_file),
                'output_dir': args.output_dir
            })
        else:
            log_warn(f"Output file not found: {output_file}")
    else:
        log_error(f"Inference failed with code {proc.returncode}")
        log_error(f"stderr: {proc.stderr.strip()}")
        # Log error to wandb
        wandb.log({'error': proc.stderr})
        sys.exit(proc.returncode)

    run.finish()


if __name__ == '__main__':
    main()
