#!/bin/bash

# Complete training pipeline for Multi-Module Reward Model

set -e  # Exit on error

# Get the directory where this script is located
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $PROJECT_DIR

echo "================================================"
echo "Multi-Module Reward Model Training Pipeline"
echo "================================================"

# Step 1: Generate synthetic data
echo ""
echo "Step 1: Generating synthetic training data..."
echo "----------------------------------------------"
python src/generate_synthetic_data.py

# Check if data was generated
if [ ! -f "dataset/training_pairs.json" ]; then
    echo "Error: Failed to generate training data"
    exit 1
fi

echo "Data generation complete!"

# Step 2: Train Stage 0 (Heads only)
echo ""
echo "Step 2: Training Stage 0 (Heads only)..."
echo "----------------------------------------------"
python src/train.py \
    --stage 0 \
    --data_path dataset/training_pairs.json \
    --batch_size 4 \
    --stage0_epochs 5 \
    --stage0_lr 3e-4

# Step 3: Train Stage 1 (Full model)
echo ""
echo "Step 3: Training Stage 1 (Full model)..."
echo "----------------------------------------------"
python src/train.py \
    --stage 1 \
    --data_path dataset/training_pairs.json \
    --batch_size 4 \
    --stage1_epochs 5 \
    --stage1_encoder_lr 1e-5 \
    --stage1_head_lr 1e-4

# Step 4: Evaluate model
echo ""
echo "Step 4: Evaluating trained model..."
echo "----------------------------------------------"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -t checkpoints/stage_1_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Warning: No Stage 1 checkpoint found, using Stage 0"
    LATEST_CHECKPOINT=$(ls -t checkpoints/stage_0_*.pt 2>/dev/null | head -1)
fi

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Using checkpoint: $LATEST_CHECKPOINT"
    python src/evaluate.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --data_path dataset/training_pairs.json \
        --plot
else
    echo "Error: No checkpoint found for evaluation"
    exit 1
fi

echo ""
echo "================================================"
echo "Pipeline Complete!"
echo "================================================"
echo ""
echo "Results saved in:"
echo "  - Checkpoints: $PROJECT_DIR/checkpoints/"
echo "  - Logs: $PROJECT_DIR/logs/"
echo "  - Plots: $PROJECT_DIR/logs/score_distributions.png"