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

# ===============================================================
# IMPORTANT: Choose ONE of the options below for FLUX_FILL_PATH
# ===============================================================

# OPTION 1: Local path (recommended for reliability)
# Download the model manually from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
# Then point to your local copy:
export FLUX_FILL_PATH="/path/to/local/FLUX.1-Fill-dev"

# OPTION 2: HuggingFace path (may require HF_TOKEN to be set)
# Uncomment the line below and comment out the local path above
# export FLUX_FILL_PATH="hf://black-forest-labs/FLUX.1-Fill-dev"

# If using OPTION 2, you may need to set your HuggingFace token:
# export HF_TOKEN="your_huggingface_token_here"

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
  
  # Check if using local path and if it exists
  if [[ "$FLUX_FILL_PATH" == "/"* ]] && [[ ! -d "$FLUX_FILL_PATH" ]]; then
    echo "  WARNING: Local directory $FLUX_FILL_PATH does not exist!"
    echo "  Please download the model from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
    echo "  and update the path in this script."
  fi
  
  # Check if using HuggingFace path and if HF_TOKEN is set
  if [[ "$FLUX_FILL_PATH" == "hf://"* ]] && [[ -z "$HF_TOKEN" ]]; then
    echo "  WARNING: Using HuggingFace path but HF_TOKEN is not set."
    echo "  You may need to set HF_TOKEN if the model is not publicly accessible."
  fi
fi

echo " PORTRAIT_MODEL_PATH = $PORTRAIT_MODEL_PATH"
echo " SUBJECT_MODEL_PATH = $SUBJECT_MODEL_PATH"
echo " LOCAL_MODEL_PATH = $LOCAL_MODEL_PATH"
echo " ACE_PLUS_FFT_MODEL = $ACE_PLUS_FFT_MODEL"

echo ""
echo "Environment variables for ACE++ have been set!"
echo "You can now run the ACE++ tools, e.g.: python demo_lora.py"