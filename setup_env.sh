#!/usr/bin/env bash
###############################################################################
#  ACE++ Environment Variables Setup Script
#  ────────────────────────────────────────────────────────────────────────────
#  ‼️  MUST be sourced, not executed:  source setup_env.sh
###############################################################################

# ── Source check ─────────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed directly!"
  echo "Usage:  source ${BASH_SOURCE[0]##*/}"
  return 1 2>/dev/null || exit 1
fi

# ── Force HOME to /app/scepter and export it ─────────────────────────────────
export HOME="/app/scepter"

# ── Optional: Persist for all future shells (requires root) ──────────────────
#    The first time this script runs as root it copies itself into
#    /etc/profile.d so every user session inherits these vars automatically.
if [[ $EUID -eq 0 ]] && [[ ! -f /etc/profile.d/ace_env.sh ]]; then
  echo "Installing ACE++ env vars system‑wide ⇒ /etc/profile.d/ace_env.sh"
  # Remove the auto‑install block to avoid recursion in /etc/profile.d copy
  awk '/^# ── Optional: Persist/{exit}1' "${BASH_SOURCE[0]}" \
      > /etc/profile.d/ace_env.sh
  chmod 644 /etc/profile.d/ace_env.sh
fi

# ── FLUX_FILL_PATH (Option 1: local path – recommended) ──────────────────────
MODEL_DIR="$HOME/ACE_plus/FLUX.1-Fill-dev"
mkdir -p "$MODEL_DIR"
export FLUX_FILL_PATH="$MODEL_DIR"

cat <<EOF

================================================================
IMPORTANT: Download the FLUX.1‑Fill‑dev model files into:
  $MODEL_DIR
(from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
================================================================

EOF

# ── LoRA / FFT model paths (default: ModelScope) ─────────────────────────────
export PORTRAIT_MODEL_PATH="ms://iic/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors"
export SUBJECT_MODEL_PATH="ms://iic/ACE_Plus@subject/comfyui_subject_lora16.safetensors"
export LOCAL_MODEL_PATH="ms://iic/ACE_Plus@local_editing/comfyui_local_lora16.safetensors"
export ACE_PLUS_FFT_MODEL="ms://iic/ACE_Plus@ace_plus_fft.safetensors.safetensors"

# ── Sanity checks ────────────────────────────────────────────────────────────
echo "FLUX_FILL_PATH     = $FLUX_FILL_PATH"
echo "PORTRAIT_MODEL_PATH= $PORTRAIT_MODEL_PATH"
echo "SUBJECT_MODEL_PATH = $SUBJECT_MODEL_PATH"
echo "LOCAL_MODEL_PATH   = $LOCAL_MODEL_PATH"
echo "ACE_PLUS_FFT_MODEL = $ACE_PLUS_FFT_MODEL"
echo
echo "ACE++ environment variables are now active in this shell."
echo "Run tools, e.g.: python demo_lora.py"
###############################################################################
