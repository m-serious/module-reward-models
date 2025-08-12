#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward_model import MultiModuleRewardModel, EmbeddingCache
from data_loader import PairwisePreferenceDataset, create_dataloaders

class Trainer:
    def __init__(
        self,
        model: MultiModuleRewardModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = None,
        log_dir: str = None,
        use_wandb: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup directories with default paths
        if checkpoint_dir is None:
            import os
            checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        if log_dir is None:
            import os
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="module-reward-model", name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def train_stage_0(
        self,
        num_epochs: int = 5,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        print("\n=== Stage 0: Training heads only (encoder frozen) ===")
        
        # Ensure encoder is frozen
        self.model.freeze_encoder()
        
        # Setup optimizer for heads and layernorm only
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"  Training: {name}")
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = self._get_scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_epochs * len(self.train_loader)
        )
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_loss, train_metrics = self._train_epoch(
                optimizer,
                scheduler,
                gradient_accumulation_steps,
                max_grad_norm,
                stage="stage_0"
            )
            
            # Validation
            if self.val_loader:
                val_loss, val_metrics = self._validate()
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("stage_0_best.pt", optimizer, scheduler)
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"stage_0_epoch_{epoch+1}.pt", optimizer, scheduler)
        
        print("Stage 0 training complete!")
        return train_metrics
    
    def train_stage_1(
        self,
        num_epochs: int = 5,
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        load_stage_0_checkpoint: bool = True
    ):
        print("\n=== Stage 1: Fine-tuning entire model ===")
        
        # Load Stage 0 checkpoint if specified
        if load_stage_0_checkpoint:
            checkpoint_path = self.checkpoint_dir / "stage_0_best.pt"
            if checkpoint_path.exists():
                print(f"Loading Stage 0 checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Stage 0 checkpoint loaded successfully!")
        
        # Unfreeze encoder
        self.model.unfreeze_encoder()
        
        # Setup optimizer with different learning rates
        encoder_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': head_lr}
        ], weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        
        print(f"  Encoder parameters: {len(encoder_params)}, lr={encoder_lr}")
        print(f"  Head parameters: {len(head_params)}, lr={head_lr}")
        
        # Learning rate scheduler
        scheduler = self._get_scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_epochs * len(self.train_loader)
        )
        
        # Reset best validation loss for Stage 1
        self.best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_loss, train_metrics = self._train_epoch(
                optimizer,
                scheduler,
                gradient_accumulation_steps,
                max_grad_norm,
                stage="stage_1"
            )
            
            # Validation
            if self.val_loader:
                val_loss, val_metrics = self._validate()
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("stage_1_best.pt", optimizer, scheduler)
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"stage_1_epoch_{epoch+1}.pt", optimizer, scheduler)
        
        print("Stage 1 training complete!")
        return train_metrics
    
    def _train_epoch(
        self,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        max_grad_norm,
        stage
    ):
        self.model.train()
        total_loss = 0.0
        module_losses = {
            'reflection': [],
            'planner': [],
            'executor': [],
            'memory': []
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Training {stage}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Process batch
            batch_loss = 0.0
            batch_module_losses = {}
            
            for module, pairs in batch['pairs'].items():
                module_loss_sum = 0.0
                
                for pair in pairs:
                    # Prepare inputs
                    task = pair['task']
                    
                    # Positive sample
                    pos_context = pair['positive']['trajectory_context']
                    pos_module_output = {module: pair['positive']['module_output']}
                    
                    # Negative sample
                    neg_context = pair['negative']['trajectory_context']
                    neg_module_output = {module: pair['negative']['module_output']}
                    
                    # Forward pass with mixed precision
                    with autocast():
                        pos_scores = self.model(pos_context, pos_module_output, [module])
                        neg_scores = self.model(neg_context, neg_module_output, [module])
                        
                        # Compute loss
                        loss, mod_losses = self.model.compute_pairwise_loss(
                            pos_scores, neg_scores, margin=0.1
                        )
                    
                    module_loss_sum += loss
                    
                    # Track module-specific losses
                    for mod, mod_loss in mod_losses.items():
                        if mod not in batch_module_losses:
                            batch_module_losses[mod] = []
                        batch_module_losses[mod].append(mod_loss)
                
                batch_loss += module_loss_sum / len(pairs)
            
            # Normalize by accumulation steps
            batch_loss = batch_loss / gradient_accumulation_steps
            
            # Backward pass with mixed precision
            self.scaler.scale(batch_loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                
                # Scheduler step
                scheduler.step()
            
            # Update metrics
            total_loss += batch_loss.item() * gradient_accumulation_steps
            for mod, losses in batch_module_losses.items():
                module_losses[mod].extend(losses)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': batch_loss.item() * gradient_accumulation_steps,
                'lr': scheduler.get_last_lr()[0]
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    f'{stage}/loss': batch_loss.item() * gradient_accumulation_steps,
                    f'{stage}/lr': scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
            
            self.global_step += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_module_losses = {
            mod: np.mean(losses) if losses else 0.0
            for mod, losses in module_losses.items()
        }
        
        return avg_loss, avg_module_losses
    
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        module_losses = {
            'reflection': [],
            'planner': [],
            'executor': [],
            'memory': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch_loss = 0.0
                
                for module, pairs in batch['pairs'].items():
                    for pair in pairs:
                        # Prepare inputs
                        task = pair['task']
                        
                        # Positive sample
                        pos_context = pair['positive']['trajectory_context']
                        pos_module_output = {module: pair['positive']['module_output']}
                        
                        # Negative sample
                        neg_context = pair['negative']['trajectory_context']
                        neg_module_output = {module: pair['negative']['module_output']}
                        
                        # Forward pass
                        pos_scores = self.model(pos_context, pos_module_output, [module])
                        neg_scores = self.model(neg_context, neg_module_output, [module])
                        
                        # Compute loss
                        loss, mod_losses = self.model.compute_pairwise_loss(
                            pos_scores, neg_scores
                        )
                        
                        batch_loss += loss
                        
                        # Track module-specific losses
                        for mod, mod_loss in mod_losses.items():
                            module_losses[mod].append(mod_loss)
                
                total_loss += batch_loss.item()
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_module_losses = {
            mod: np.mean(losses) if losses else 0.0
            for mod, losses in module_losses.items()
        }
        
        return avg_loss, avg_module_losses
    
    def _get_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, filename, optimizer, scheduler):
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Module Reward Model")
    parser.add_argument('--stage', type=str, choices=['0', '1', 'both'], default='both',
                        help='Training stage: 0 (heads only), 1 (full model), or both')
    parser.add_argument('--data_path', type=str,
                        default='dataset/training_pairs.json')
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--stage0_epochs', type=int, default=5)
    parser.add_argument('--stage1_epochs', type=int, default=5)
    parser.add_argument('--stage0_lr', type=float, default=3e-4)
    parser.add_argument('--stage1_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--stage1_head_lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load checkpoint before training')
    
    args = parser.parse_args()
    
    # Create model
    print("Initializing model...")
    model = MultiModuleRewardModel(
        encoder_name="Qwen/Qwen3-Embedding-0.6B",
        emb_dim=1024,
        hidden_dim=2048,
        freeze_encoder=True
    )
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_path=args.data_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=args.use_wandb
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Training
    if args.stage in ['0', 'both']:
        trainer.train_stage_0(
            num_epochs=args.stage0_epochs,
            learning_rate=args.stage0_lr
        )
    
    if args.stage in ['1', 'both']:
        trainer.train_stage_1(
            num_epochs=args.stage1_epochs,
            encoder_lr=args.stage1_encoder_lr,
            head_lr=args.stage1_head_lr,
            load_stage_0_checkpoint=(args.stage == 'both')
        )
    
    print("Training complete!")

if __name__ == "__main__":
    main()