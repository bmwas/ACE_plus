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

3a. Text-to-image generation (default):
   python3.10 run_inference.py \
     --mode text2image \
     --task_type subject \
     --instruction "A doodle of a cat" \
     --output_h 512 \
     --output_w 512 \
     --seed 42 \
     --input_reference_image ./assets/samples/control/resuzed_balnk.webp \
     --output_dir ./examples/output_images

3b. Multi-step image editing:
   python3.10 run_inference.py \
     --mode edit \
     --input_image ./examples/output_images/generated_image_0_5889fc14417243d86aa7.png \
     --edit_instructions "remove all words and pencils" \
     --output_dir ./examples/output_images

3c. python3.10 run_inference.py --mode edit \
  --input_image ./examples/output_images/generated_image_0_5889fc14417243d86aa7.png \
  --edit_instructions "remove all words and pencils" \
  --output_dir ./examples/output_images

Options:
  --mode                   text2image (default) or edit
  --task_type              portrait, subject, local_editing (default: subject)
  --instruction            prompt text (for text2image)
  --output_h, --output_w   height/width in pixels
  --seed                   random seed for reproducibility
  --input_reference_image  path to reference image
  --output_dir             directory to save outputs
  --input_image            path to initial image for editing (edit mode)
  --edit_instructions      one or more edit instructions (edit mode)

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

import wandb
from PIL import Image

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
    parser.add_argument('--mode', choices=['text2image', 'edit'], default='text2image',
                        help='Choose inference mode: text2image or edit (default: text2image)')
    parser.add_argument('--task_type', choices=['portrait', 'subject', 'local_editing'], default='subject',
                        help='Task type: portrait, subject, local_editing')
    parser.add_argument('--instruction', default='A beautiful landscape with mountains, clear blue sky, and a lake',
                        help='Text prompt for generation (text2image mode only)')
    parser.add_argument('--output_h', type=int, default=512, help='Output height')
    parser.add_argument('--output_w', type=int, default=512, help='Output width')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input_reference_image', default='./assets/samples/control/resuzed_balnk.webp',
                        help='Path to reference image')
    parser.add_argument('--output_dir', default='./examples/output_images',
                        help='Directory to save outputs')
    # Edit mode arguments
    parser.add_argument('--input_image', default=None, help='Path to initial image for editing (edit mode)')
    parser.add_argument('--edit_instructions', nargs='*', default=None,
                        help='Edit instructions (one or more, edit mode)')
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
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log_info(f"GPU available: {name}")
        else:
            log_warn("No GPU available.")
    except Exception as e:
        log_warn(f"Error checking GPU: {e}")

def run_text2image(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"generated_{args.task_type}_{ts}.png")
    os.environ['PORTRAIT_MODEL_PATH'] = 'ms://iic/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors'
    os.environ['SUBJECT_MODEL_PATH'] = 'ms://iic/ACE_Plus@subject/comfyui_subject_lora16.safetensors'
    os.environ['LOCAL_MODEL_PATH'] = 'ms://iic/ACE_Plus@local_editing/comfyui_local_lora16.safetensors'
    log_info("Starting text-to-image inference process")
    log_debug(f"Task Type: {args.task_type}")
    log_debug(f"Instruction: {args.instruction}")
    log_debug(f"Output Size: {args.output_h}x{args.output_w}")
    log_debug(f"Seed: {args.seed}")
    log_debug(f"Reference Image: {args.input_reference_image}")
    log_debug(f"Output Dir: {args.output_dir}")
    log_debug(f"Output File: {output_file}")
    check_dependencies()
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
    else:
        log_warn("WANDB_API_KEY not found in environment; skipping wandb login.")
    run = wandb.init(project='ace_plus_inference', config={
        'mode': 'text2image',
        'task_type': args.task_type,
        'instruction': args.instruction,
        'output_h': args.output_h,
        'output_w': args.output_w,
        'seed': args.seed,
        'input_reference_image': args.input_reference_image,
        'output_dir': args.output_dir,
    })
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
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if proc.returncode == 0:
        log_info(f"Inference completed successfully in {elapsed:.2f}s")
        if os.path.isfile(output_file):
            size = os.path.getsize(output_file)
            log_debug(f"Generated image size: {size} bytes")
            log_info(f"Image generated: {output_file}")
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
        wandb.log({'error': proc.stderr})
        sys.exit(proc.returncode)
    run.finish()

def run_edit(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.input_image or not os.path.isfile(args.input_image):
        log_error("For edit mode, --input_image must be provided and must exist.")
        sys.exit(1)
    if not args.edit_instructions or len(args.edit_instructions) == 0:
        log_error("For edit mode, provide at least one --edit_instructions value.")
        sys.exit(1)
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
    else:
        log_warn("WANDB_API_KEY not found in environment; skipping wandb login.")
    run = wandb.init(project='ace_plus_inference', config={
        'mode': 'edit',
        'input_image': args.input_image,
        'edit_instructions': args.edit_instructions,
        'output_dir': args.output_dir,
        'output_h': args.output_h,
        'output_w': args.output_w,
        'seed': args.seed,
        'task_type': args.task_type
    })
    # Iterative editing loop
    current_image_path = args.input_image
    for idx, instruction in enumerate(args.edit_instructions):
        output_file = os.path.join(
            args.output_dir,
            f"editstep{idx+1}_{os.path.basename(current_image_path).split('.')[0]}_{ts}.png"
        )
        log_info(f"[Edit Step {idx+1}] Instruction: {instruction}")
        log_debug(f"Input image: {current_image_path}")
        log_debug(f"Output file: {output_file}")
        cmd = [
            sys.executable, 'infer_lora.py',
            '--input_image', current_image_path,
            '--instruction', instruction,
            '--output_h', str(args.output_h),
            '--output_w', str(args.output_w),
            '--seed', str(args.seed),
            '--task_type', args.task_type,
            '--save_path', output_file
        ]
        log_info(f"Command: {' '.join(cmd)}")
        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        if proc.returncode == 0:
            log_info(f"[Edit Step {idx+1}] Completed in {elapsed:.2f}s")
            if os.path.isfile(output_file):
                size = os.path.getsize(output_file)
                log_debug(f"[Edit Step {idx+1}] Output image size: {size} bytes")
                wandb.log({
                    f'editstep{idx+1}_instruction': instruction,
                    f'editstep{idx+1}_image': wandb.Image(output_file),
                    'output_dir': args.output_dir,
                    'step': idx+1,
                    'elapsed_time': elapsed
                })
                current_image_path = output_file
            else:
                log_warn(f"[Edit Step {idx+1}] Output file not found: {output_file}")
        else:
            log_error(f"[Edit Step {idx+1}] Failed with code {proc.returncode}")
            log_error(f"stderr: {proc.stderr.strip()}")
            wandb.log({f'editstep{idx+1}_error': proc.stderr})
            run.finish()
            sys.exit(proc.returncode)
    run.finish()
    log_info("All edit steps completed.")

def main():
    args = parse_args()
    if args.mode == 'text2image':
        run_text2image(args)
    elif args.mode == 'edit':
        run_edit(args)
    else:
        log_error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()
