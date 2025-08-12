#!/usr/bin/env python3
"""
Basic test script to verify the implementation works without errors
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import json
from pathlib import Path

def test_data_generation():
    print("\n1. Testing data generation...")
    from generate_synthetic_data import generate_dataset, save_dataset
    
    # Generate small test dataset
    dataset = generate_dataset()
    assert len(dataset) == 64, f"Expected 64 samples, got {len(dataset)}"
    
    # Check structure
    sample = dataset[0]
    required_keys = ['task', 'trajectory_full_context', 'module_k_all_rounds', 
                     'outcome', 'preference', 'target_module', 'pair_id', 'is_positive']
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    print("   ✓ Data generation test passed")
    return dataset

def test_model_creation():
    print("\n2. Testing model creation...")
    from reward_model import MultiModuleRewardModel
    
    # Create model with smaller dimensions for testing
    model = MultiModuleRewardModel(
        encoder_name="Qwen/Qwen3-Embedding-0.6B",
        emb_dim=1024,
        hidden_dim=512,  # Smaller for testing
        freeze_encoder=True
    )
    
    # Check modules
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'module_heads')
    assert len(model.module_heads) == 4
    
    # Check if encoder is frozen
    for param in model.encoder.parameters():
        assert not param.requires_grad, "Encoder should be frozen"
    
    print("   ✓ Model creation test passed")
    return model

def test_forward_pass(model):
    print("\n3. Testing forward pass...")
    
    # Create dummy inputs
    trajectory_context = "This is a test trajectory context with multiple rounds of interaction."
    module_outputs = {
        'reflection': "Test reflection output",
        'planner': "Test planner output",
        'executor': "Test executor output",
        'memory': "Test memory output"
    }
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        scores = model(trajectory_context, module_outputs)
    
    # Check output
    assert len(scores) == 4, f"Expected 4 scores, got {len(scores)}"
    for module, score in scores.items():
        assert isinstance(score, torch.Tensor), f"Score for {module} is not a tensor"
        assert score.shape == torch.Size([]), f"Score shape incorrect for {module}"
    
    print("   ✓ Forward pass test passed")

def test_loss_computation(model):
    print("\n4. Testing loss computation...")
    
    # Create dummy scores
    positive_scores = {
        'reflection': torch.tensor(0.8),
        'planner': torch.tensor(0.7),
        'executor': torch.tensor(0.9),
        'memory': torch.tensor(0.6)
    }
    
    negative_scores = {
        'reflection': torch.tensor(0.3),
        'planner': torch.tensor(0.4),
        'executor': torch.tensor(0.2),
        'memory': torch.tensor(0.5)
    }
    
    # Compute loss
    loss, module_losses = model.compute_pairwise_loss(positive_scores, negative_scores)
    
    # Check loss
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert loss.shape == torch.Size([]), "Loss shape incorrect"
    assert loss.item() > 0, "Loss should be positive"
    assert len(module_losses) == 4, f"Expected 4 module losses, got {len(module_losses)}"
    
    print("   ✓ Loss computation test passed")

def test_data_loader():
    print("\n5. Testing data loader...")
    from data_loader import PairwisePreferenceDataset, create_dataloaders
    
    # First ensure data exists
    data_path = Path("dataset/training_pairs.json")
    if not data_path.exists():
        print("   Creating test data...")
        from generate_synthetic_data import generate_dataset, save_dataset
        dataset = generate_dataset()
        save_dataset(dataset, "dataset")
    
    # Create dataset
    dataset = PairwisePreferenceDataset(str(data_path))
    assert len(dataset) > 0, "Dataset is empty"
    
    # Test single sample
    sample = dataset[0]
    required_keys = ['pair_id', 'target_module', 'task', 'positive', 'negative']
    for key in required_keys:
        assert key in sample, f"Missing key in sample: {key}"
    
    # Create dataloader
    train_loader, _ = create_dataloaders(
        train_path=str(data_path),
        batch_size=2,
        num_workers=0  # Use 0 for testing
    )
    
    # Test batch
    for batch in train_loader:
        assert 'modules' in batch
        assert 'pairs' in batch
        assert 'batch_size' in batch
        break  # Just test first batch
    
    print("   ✓ Data loader test passed")

def test_training_step():
    print("\n6. Testing training step...")
    from reward_model import MultiModuleRewardModel
    from data_loader import create_dataloaders
    
    # Ensure data exists
    data_path = Path("dataset/training_pairs.json")
    if not data_path.exists():
        from generate_synthetic_data import generate_dataset, save_dataset
        dataset = generate_dataset()
        save_dataset(dataset, "dataset")
    
    # Create model
    model = MultiModuleRewardModel(
        encoder_name="Qwen/Qwen3-Embedding-0.6B",
        emb_dim=1024,
        hidden_dim=512,
        freeze_encoder=True
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # Create dataloader
    train_loader, _ = create_dataloaders(
        train_path=str(data_path),
        batch_size=2,
        num_workers=0
    )
    
    # Training step
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        total_loss = 0
        
        for module, pairs in batch['pairs'].items():
            for pair in pairs:
                # Forward pass
                pos_context = pair['positive']['trajectory_context']
                pos_module_output = {module: pair['positive']['module_output']}
                pos_scores = model(pos_context, pos_module_output, [module])
                
                neg_context = pair['negative']['trajectory_context']
                neg_module_output = {module: pair['negative']['module_output']}
                neg_scores = model(neg_context, neg_module_output, [module])
                
                # Compute loss
                loss, _ = model.compute_pairwise_loss(pos_scores, neg_scores)
                total_loss = total_loss + loss
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            optimizer.step()
        
        break  # Just test one batch
    
    print("   ✓ Training step test passed")

def main():
    print("="*50)
    print("Running Basic Tests for Module Reward Model")
    print("="*50)
    
    try:
        # Test data generation
        dataset = test_data_generation()
        
        # Test model
        model = test_model_creation()
        test_forward_pass(model)
        test_loss_computation(model)
        
        # Test data loading
        test_data_loader()
        
        # Test training
        test_training_step()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("="*50)
        print("\nThe implementation is working correctly.")
        print("You can now run the full training pipeline with:")
        print("  bash run_pipeline.sh")
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"TEST FAILED: {e}")
        print("="*50)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()