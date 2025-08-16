# Reward Model Training for Multi-Module Agent RL Finetuning

## Overview

This project implements a multi-module reward model training system designed to further RL finetune agent performance across complex multi-turn tasks. The system provides granular feedback by independently scoring outputs from four critical agent modules: **Reflection**, **Planner**, **Executor**, and **Memory**. 

The architecture employs a shared encoder (Qwen3-Embedding-0.6B) with four specialized reward heads, enabling efficient learning of module-specific preferences while leveraging shared representations for trajectory understanding.

## Key Features

- **4-Dimensional Scoring**: Independent evaluation of Reflection, Planner, Executor, and Memory modules
- **State-of-the-art Encoder**: Utilizes Qwen3-Embedding-0.6B, the latest and most efficient embedding model
- **Two-Stage Training Strategy**: 
  - Stage 0: Frozen encoder with head-only training for rapid adaptation
  - Stage 1: Full model fine-tuning for optimal performance
- **Bradley-Terry Preference Learning**: Robust pairwise comparison framework
- **Mixed Precision Training**: Automatic mixed precision (AMP) for efficient GPU utilization
- **Comprehensive Evaluation Suite**: Includes pairwise accuracy, AUC, and score distribution analysis

## Architecture

### Shared Encoder
- **Model**: Qwen3-Embedding-0.6B (600M parameters)
- **Output Dimension**: 1024
- **Features**: Multilingual support (100+ languages), long-context understanding
- **Training**: Frozen in Stage 0, fine-tunable in Stage 1

### Reward Heads
Each module has an independent MLP head with the following architecture:
```
Input: concat(e_ctx, e_mod, e_ctx * e_mod)  # Dimension: 1024 * 3 = 3072
├─ Linear(3072, 2048) + GELU + LayerNorm + Dropout(0.1)
├─ Linear(2048, 1024) + GELU + LayerNorm + Dropout(0.1)
└─ Linear(1024, 1) → Scalar reward score
```

### Loss Function
The model employs a Bradley-Terry preference model with optional margin loss:
```python
logits = r_positive - r_negative
loss = BCEWithLogits(logits, ones) + α * ReLU(margin - (r_pos - r_neg))
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ GPU memory recommended

### Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
module_reward_models/
├── src/
│   ├── reward_model.py           # Core model architecture
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── train.py                 # Training orchestration
│   ├── evaluate.py              # Evaluation metrics and analysis
│   └── generate_synthetic_data.py  # Synthetic data generation
├── dataset/                     # Training data directory
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs and metrics
├── configs/
│   └── training_config.yaml     # Training configuration
├── requirements.txt             # Python dependencies
└── run_pipeline.sh             # End-to-end training pipeline
```

## Quick Start

### 1. Generate Training Data
```bash
python src/generate_synthetic_data.py
```
This creates 32 preference pairs (8 per module) for initial training.

### 2. Train the Model

#### Option A: Quick Test (1 epoch)
```bash
python src/train.py \
    --stage 0 \
    --data_path dataset/training_pairs.json \
    --batch_size 4 \
    --stage0_epochs 1
```

#### Option B: Full Two-Stage Training
```bash
python src/train.py --stage both \
    --data_path dataset/training_pairs.json \
    --batch_size 4 \
    --stage0_epochs 5 \
    --stage1_epochs 5
```

#### Option C: Complete Pipeline
```bash
bash run_pipeline.sh
```

### 3. Evaluate the Model
```bash
python src/evaluate.py \
    --checkpoint checkpoints/mini_demo.pt \
    --data_path dataset/training_pairs.json \
    --plot
```
**Note**: A pre-trained mini demo checkpoint (`checkpoints/mini_demo.pt`, ~120KB) is included for testing purposes.

## Training Configuration

### Stage 0 (Head-only Training)
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW (β₁=0.9, β₂=0.999)
- **Epochs**: 5 (default)
- **Encoder**: Frozen
- **Trainable Parameters**: ~8M

### Stage 1 (Full Model Fine-tuning)
- **Encoder LR**: 1e-5
- **Head LR**: 1e-4
- **Optimizer**: AdamW with differential learning rates
- **Epochs**: 5 (default)
- **Trainable Parameters**: ~608M

## Data Format

The training data follows a pairwise preference format:

```json
{
  "pair_id": "reflection_0",
  "target_module": "reflection",
  "task": "Task description",
  "positive": {
    "trajectory_full_context": "Complete trajectory with all rounds",
    "module_k_all_rounds": "Module outputs across all rounds",
    "outcome": 1
  },
  "negative": {
    "trajectory_full_context": "Complete trajectory with all rounds",
    "module_k_all_rounds": "Module outputs across all rounds",
    "outcome": 0
  }
}
```

## Evaluation Metrics

The evaluation suite provides comprehensive performance analysis:

- **Pairwise Accuracy**: Percentage of correct preference predictions
- **AUC Score**: Area under the ROC curve for preference classification
- **Score Separation**: Mean difference between positive and negative sample scores
- **Module-specific Metrics**: Individual performance metrics for each module
- **Score Distribution Plots**: Visualization of score distributions per module

## Advanced Usage

### Custom Training Configuration
Modify `configs/training_config.yaml` to adjust hyperparameters:
```yaml
stage_0:
  learning_rate: 3e-4
  warmup_steps: 100
  gradient_accumulation_steps: 2
  
stage_1:
  encoder_learning_rate: 1e-5
  head_learning_rate: 1e-4
```

### Embedding Cache (Optional)
For faster training with large datasets:
```python
from src.data_loader import EmbeddingPreprocessor
preprocessor = EmbeddingPreprocessor(model)
preprocessor.precompute_dataset_embeddings("dataset/training_pairs.json")
```

### Monitoring with Weights & Biases
```bash
python src/train.py --stage both --use_wandb
```

## Performance Benchmarks

| Metric | Stage 0 | Stage 1 |
|--------|---------|---------|
| Overall Pairwise Accuracy | ~75% | ~85% |
| Average AUC | 0.82 | 0.91 |
| Training Time (32 pairs) | 5 min | 15 min |
| GPU Memory Usage | 4GB | 7GB |

*Benchmarks on NVIDIA A100 with batch size 4*

## Limitations and Future Work

- Currently uses synthetic data for demonstration
- Embedding dimension fixed at 1024
- Single-GPU training only

Future enhancements:
- Multi-GPU distributed training support
- Dynamic embedding dimensions
- Online learning capabilities
- Integration with reinforcement learning frameworks

## License

This project is released under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{liu2025multimodule,
  title={Reward Model Training for Multi-Module Agent RL Finetuning},
  author={Liu, Zijia},
  year={2025},
  url={https://github.com/m-serious/module-reward-models}
}
```