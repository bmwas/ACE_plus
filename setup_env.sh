#!/bin/bash

# ACE++ Environment Variables Setup Script
# This script exports all necessary environment variables for running ACE++ models

# ===============================================================
# IMPORTANT: This script must be sourced, not executed directly!
# Run it using: source setup_env.sh
# ===============================================================

# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed directly!"
  echo "Please run it using: source setup_env.sh"
  exit 1
fi

# Base model - FLUX.1-Fill-dev (Required for all configurations)
# Option 1: Local path (uncomment and set your local path)
# export FLUX_FILL_PATH="/path/to/FLUX.1-Fill-dev"

# Option 2: Hugging Face path (recommended if you don't have the model downloaded)
export FLUX_FILL_PATH="hf://black-forest-labs/FLUX.1-Fill-dev"

# LoRA model paths
# Option 1: ModelScope paths (default)
export PORTRAIT_MODEL_PATH="ms://iic/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors"
export SUBJECT_MODEL_PATH="ms://iic/ACE_Plus@subject/comfyui_subject_lora16.safetensors"
export LOCAL_MODEL_PATH="ms://iic/ACE_Plus@local_editing/comfyui_local_lora16.safetensors"

# Option 2: HuggingFace paths (uncomment to use)
# export PORTRAIT_MODEL_PATH="hf://ali-vilab/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors"
# export SUBJECT_MODEL_PATH="hf://ali-vilab/ACE_Plus@subject/comfyui_subject_lora16.safetensors"
# export LOCAL_MODEL_PATH="hf://ali-vilab/ACE_Plus@local_editing/comfyui_local_lora16.safetensors"

# FFT model path (only needed for FFT models)
export ACE_PLUS_FFT_MODEL="ms://iic/ACE_Plus@ace_plus_fft.safetensors.safetensors"

# Verify environment variables were set properly
if [[ -z "$FLUX_FILL_PATH" ]]; then
  echo "WARNING: FLUX_FILL_PATH is not set properly!"
else
  echo " FLUX_FILL_PATH = $FLUX_FILL_PATH"
fi

echo " PORTRAIT_MODEL_PATH = $PORTRAIT_MODEL_PATH"
echo " SUBJECT_MODEL_PATH = $SUBJECT_MODEL_PATH"
echo " LOCAL_MODEL_PATH = $LOCAL_MODEL_PATH"
echo " ACE_PLUS_FFT_MODEL = $ACE_PLUS_FFT_MODEL"

echo ""
echo "Environment variables for ACE++ have been set!"
echo "You can now run the ACE++ tools, e.g.: python demo_lora.py"