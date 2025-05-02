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

# OPTION 1: Local path approach (recommended due to HuggingFace access issues)
# Create a directory for models and set the path to it:
MODEL_DIR="$HOME/ACE_plus/FLUX.1-Fill-dev"
mkdir -p "$MODEL_DIR"
export FLUX_FILL_PATH="$MODEL_DIR"

echo ""
echo "============================================================="
echo "IMPORTANT: You need to download the FLUX.1-Fill-dev model files"
echo "and place them in: $MODEL_DIR"
echo "Download from: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
echo "============================================================="
echo ""

# OPTION 2: Using HuggingFace path (not recommended unless you have HF authentication)
# Uncomment the line below and comment out the local path section above if you want to try this
# export FLUX_FILL_PATH="hf://black-forest-labs/FLUX.1-Fill-dev"
# export HF_TOKEN="your_huggingface_token_here"  # You may need to set this

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
  
  # Check if using local path and if it has model files
  if [[ "$FLUX_FILL_PATH" != "hf://"* ]]; then
    if [[ ! -d "$FLUX_FILL_PATH" ]]; then
      echo "  WARNING: Local directory $FLUX_FILL_PATH does not exist!"
    else
      # Check if directory is empty
      if [ -z "$(ls -A "$FLUX_FILL_PATH")" ]; then
        echo "  WARNING: Directory $FLUX_FILL_PATH exists but is empty!"
        echo "  Please download the model files from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
      fi
    fi
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