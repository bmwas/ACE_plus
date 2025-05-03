#!/bin/bash

# run_inference.sh - Demo script for ACE_plus LoRA model inference
# This script demonstrates text-to-image generation using ACE_plus LoRA models
# with comprehensive logging at every step

# Set up logging functions with different levels
log_info() {
    echo -e "\033[0;32m[INFO] $(date '+%Y-%m-%d %H:%M:%S')\033[0m: $1"
}

log_warn() {
    echo -e "\033[0;33m[WARNING] $(date '+%Y-%m-%d %H:%M:%S')\033[0m: $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR] $(date '+%Y-%m-%d %H:%M:%S')\033[0m: $1"
}

log_debug() {
    echo -e "\033[0;36m[DEBUG] $(date '+%Y-%m-%d %H:%M:%S')\033[0m: $1"
}

# Display script banner
echo "=========================================================="
log_info "ACE_plus LoRA Model Inference Script"
echo "=========================================================="

# Set default parameters
TASK_TYPE="subject"  # Options: portrait, subject, local_editing
OUTPUT_DIR="./examples/output_images"
SEED=42
OUTPUT_H=512  # Using smaller resolution for faster inference
OUTPUT_W=512
INSTRUCTION="A beautiful landscape with mountains, clear blue sky, and a lake"
INPUT_REFERENCE_IMAGE="./assets/samples/control/resuzed_balnk.webp"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_type)
            TASK_TYPE="$2"
            # Convert to lowercase if needed
            TASK_TYPE=$(echo "$TASK_TYPE" | tr '[:upper:]' '[:lower:]')
            shift 2
            ;;
        --instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        --output_h)
            OUTPUT_H="$2"
            shift 2
            ;;
        --output_w)
            OUTPUT_W="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --input_reference_image)
            INPUT_REFERENCE_IMAGE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --task_type     Model type: portrait, subject, or local_editing (default: subject)"
            echo "  --instruction   Text prompt for generation (default: landscape scene)"
            echo "  --output_h      Output height (default: 512)"
            echo "  --output_w      Output width (default: 512)"
            echo "  --seed          Random seed for reproducibility (default: 42)"
            echo "  --input_reference_image Path to reference image (default: ./assets/samples/control/resuzed_balnk.webp)"
            exit 0
            ;;
        *)
            log_error "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    log_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Set up output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/generated_${TASK_TYPE}_${TIMESTAMP}.png"

# Set environment variables for model paths
# This is critical for the model to find the correct LoRA files
log_info "Setting up environment variables for models"
export PORTRAIT_MODEL_PATH="ms://iic/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors"
export SUBJECT_MODEL_PATH="ms://iic/ACE_Plus@subject/comfyui_subject_lora16.safetensors"
export LOCAL_MODEL_PATH="ms://iic/ACE_Plus@local_editing/comfyui_local_lora16.safetensors"

log_info "Starting inference process with the following parameters:"
log_debug "  Task Type:    $TASK_TYPE"
log_debug "  Instruction:  $INSTRUCTION"
log_debug "  Output Size:  ${OUTPUT_H}x${OUTPUT_W}"
log_debug "  Seed:         $SEED"
log_debug "  Output Path:  $OUTPUT_FILE"
log_debug "  Reference Image: $INPUT_REFERENCE_IMAGE"
log_debug "  Portrait Model Path: $PORTRAIT_MODEL_PATH"
log_debug "  Subject Model Path:  $SUBJECT_MODEL_PATH"
log_debug "  Local Edit Model Path: $LOCAL_MODEL_PATH"

# Check if python environment is properly set up
if ! command -v python3.10 &> /dev/null; then
    log_error "Python 3.10 is not installed or not in PATH"
    exit 1
fi

log_info "Checking dependencies..."
# Note: In a real implementation, you might want to check specific packages with pip list | grep package_name

# Print GPU information if available
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU information:"
    nvidia-smi
else
    log_warn "nvidia-smi not found. No GPU information available."
fi

# If no reference image is provided and the task requires it, use the default blank image
if [ -z "$INPUT_REFERENCE_IMAGE" ]; then
    INPUT_REFERENCE_IMAGE="./assets/samples/control/resuzed_balnk.webp"
    log_info "No reference image provided. Using default: $INPUT_REFERENCE_IMAGE"
fi

# Run the inference
log_info "Running inference with LoRA model for task type: $TASK_TYPE"
log_info "Command: python3.10 infer_lora.py --instruction \"$INSTRUCTION\" --output_h $OUTPUT_H --output_w $OUTPUT_W --seed $SEED --task_type $TASK_TYPE --input_reference_image \"$INPUT_REFERENCE_IMAGE\" --save_path $OUTPUT_FILE"

# Execute the python script with timing information
log_info "Starting inference process..."
START_TIME=$(date +%s)

python3.10 infer_lora.py \
    --instruction "$INSTRUCTION" \
    --output_h $OUTPUT_H \
    --output_w $OUTPUT_W \
    --seed $SEED \
    --task_type $TASK_TYPE \
    --input_reference_image "$INPUT_REFERENCE_IMAGE" \
    --save_path $OUTPUT_FILE

RESULT=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $RESULT -eq 0 ]; then
    log_info "Inference completed successfully in ${ELAPSED} seconds"
    
    # Check if the output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        log_info "Image generated successfully: $OUTPUT_FILE"
        
        # Get file details
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        log_debug "Generated image size: $FILE_SIZE"
        
        echo "=========================================================="
        log_info "Generation successful!"
        echo "Image saved to: $OUTPUT_FILE"
        echo "=========================================================="
    else
        log_warn "The output file was not created: $OUTPUT_FILE"
    fi
else
    log_error "Inference failed with exit code $RESULT"
    echo "=========================================================="
    log_error "Generation failed!"
    echo "=========================================================="
    exit 1
fi

# Print advanced usage examples
log_info "Advanced usage examples:"
echo "  1. Portrait generation:"
echo "     ./run_inference.sh --task_type portrait --instruction \"A woman with long blonde hair and blue eyes, professional portrait\" --input_reference_image ./assets/samples/control/resuzed_balnk.webp"
echo
echo "  2. Subject-driven generation:"
echo "     ./run_inference.sh --task_type subject --instruction \"Display the logo on a billboard in a city street\" --input_reference_image ./assets/samples/control/resuzed_balnk.webp"
echo
echo "  3. With reference image (requires additional parameters in script):"
echo "     ./run_inference.sh --task_type subject --instruction \"Any prompt\" --input_reference_image ./path/to/your/reference.jpg"

exit 0
