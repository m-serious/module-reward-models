#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import numpy as np

class RewardHead(nn.Module):
    def __init__(self, emb_dim: int = 1024, hidden_dim: int = 2048):
        super().__init__()
        # Input: concat(ctx, module, ctx*module)
        input_dim = emb_dim * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-5),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2, eps=1e-5),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, e_ctx: torch.Tensor, e_mod: torch.Tensor) -> torch.Tensor:
        # Compute interaction features
        x = torch.cat([e_ctx, e_mod, e_ctx * e_mod], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score

class MultiModuleRewardModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        emb_dim: int = 1024,  # Qwen3-Embedding-0.6B output dimension
        hidden_dim: int = 2048,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        # Load encoder
        self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Shared LayerNorm for all embeddings
        self.layernorm = nn.LayerNorm(emb_dim, eps=1e-5)
        
        # Module-specific reward heads
        self.module_heads = nn.ModuleDict({
            'reflection': RewardHead(emb_dim, hidden_dim),
            'planner': RewardHead(emb_dim, hidden_dim),
            'executor': RewardHead(emb_dim, hidden_dim),
            'memory': RewardHead(emb_dim, hidden_dim)
        })
        
        # Freeze encoder if specified
        if freeze_encoder:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad() if not self.encoder.training else torch.enable_grad():
            outputs = self.encoder(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def forward(
        self,
        trajectory_context: str,
        module_outputs: Dict[str, str],
        target_modules: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode trajectory context
        e_ctx = self.encode_text(trajectory_context)
        e_ctx = self.layernorm(e_ctx)
        
        # If no target modules specified, use all
        if target_modules is None:
            target_modules = list(self.module_heads.keys())
        
        scores = {}
        for module in target_modules:
            if module in module_outputs:
                # Encode module output
                e_mod = self.encode_text(module_outputs[module])
                e_mod = self.layernorm(e_mod)
                
                # Compute reward score
                score = self.module_heads[module](e_ctx, e_mod)
                scores[module] = score
        
        return scores
    
    def compute_pairwise_loss(
        self,
        positive_scores: Dict[str, torch.Tensor],
        negative_scores: Dict[str, torch.Tensor],
        margin: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        module_losses = {}
        
        for module in positive_scores.keys():
            if module in negative_scores:
                # Bradley-Terry model: P(A > B) = sigmoid(r_A - r_B)
                logits = positive_scores[module] - negative_scores[module]
                
                # Binary cross-entropy loss
                labels = torch.ones_like(logits)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                
                # Optional margin-based loss
                if margin > 0:
                    margin_loss = F.relu(margin - (positive_scores[module] - negative_scores[module]))
                    loss = loss + 0.1 * margin_loss.mean()
                
                total_loss = total_loss + loss
                module_losses[module] = loss.item()
        
        # Average loss across modules
        num_modules = len(module_losses)
        if num_modules > 0:
            total_loss = total_loss / num_modules
        
        return total_loss, module_losses
    
    def predict_preferences(
        self,
        trajectory_context: str,
        module_outputs_a: Dict[str, str],
        module_outputs_b: Dict[str, str]
    ) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            scores_a = self.forward(trajectory_context, module_outputs_a)
            scores_b = self.forward(trajectory_context, module_outputs_b)
            
            preferences = {}
            for module in scores_a.keys():
                if module in scores_b:
                    # Probability that A is preferred over B
                    logit = scores_a[module] - scores_b[module]
                    prob = torch.sigmoid(logit).item()
                    preferences[module] = prob
            
            return preferences

class EmbeddingCache:
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            import os
            # Use cache directory relative to project root
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, text_hash: str) -> str:
        import os
        return os.path.join(self.cache_dir, f"{text_hash}.pt")
    
    def compute_hash(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def save_embedding(self, text: str, embedding: torch.Tensor):
        text_hash = self.compute_hash(text)
        cache_path = self.get_cache_path(text_hash)
        torch.save(embedding.cpu(), cache_path)
    
    def load_embedding(self, text: str) -> Optional[torch.Tensor]:
        text_hash = self.compute_hash(text)
        cache_path = self.get_cache_path(text_hash)
        import os
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        return None
    
    def clear_cache(self):
        import os
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)