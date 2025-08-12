#!/usr/bin/env python3
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
import random
from pathlib import Path
import numpy as np

class PairwisePreferenceDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_context_length: int = 2048,
        max_module_length: int = 512,
        shuffle_pairs: bool = True,
        cache_embeddings: bool = False
    ):
        self.max_context_length = max_context_length
        self.max_module_length = max_module_length
        self.cache_embeddings = cache_embeddings
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        
        if shuffle_pairs:
            random.shuffle(self.pairs)
        
        # Build module index
        self.module_to_pairs = {
            'reflection': [],
            'planner': [],
            'executor': [],
            'memory': []
        }
        
        for idx, pair in enumerate(self.pairs):
            module = pair['target_module']
            self.module_to_pairs[module].append(idx)
        
        print(f"Loaded {len(self.pairs)} pairs from {data_path}")
        for module, indices in self.module_to_pairs.items():
            print(f"  {module}: {len(indices)} pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        
        # Truncate texts if needed
        task = pair['task'][:self.max_context_length]
        
        positive_context = pair['positive']['trajectory_full_context'][:self.max_context_length]
        positive_module = pair['positive']['module_k_all_rounds'][:self.max_module_length]
        
        negative_context = pair['negative']['trajectory_full_context'][:self.max_context_length]
        negative_module = pair['negative']['module_k_all_rounds'][:self.max_module_length]
        
        return {
            'pair_id': pair['pair_id'],
            'target_module': pair['target_module'],
            'task': task,
            'positive': {
                'trajectory_context': positive_context,
                'module_output': positive_module,
                'outcome': pair['positive']['outcome']
            },
            'negative': {
                'trajectory_context': negative_context,
                'module_output': negative_module,
                'outcome': pair['negative']['outcome']
            }
        }
    
    def get_module_batch(self, module: str, batch_size: int) -> List[Dict[str, Any]]:
        indices = self.module_to_pairs.get(module, [])
        if not indices:
            return []
        
        sampled_indices = random.sample(indices, min(batch_size, len(indices)))
        return [self[idx] for idx in sampled_indices]

class DataCollator:
    def __init__(self, combine_contexts: bool = True):
        self.combine_contexts = combine_contexts
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Group by module
        module_groups = {}
        for item in batch:
            module = item['target_module']
            if module not in module_groups:
                module_groups[module] = []
            module_groups[module].append(item)
        
        # Prepare batch data
        batch_data = {
            'modules': list(module_groups.keys()),
            'pairs': module_groups,
            'batch_size': len(batch)
        }
        
        return batch_data

def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    # Create datasets
    train_dataset = PairwisePreferenceDataset(train_path, shuffle_pairs=shuffle)
    
    val_dataset = None
    if val_path and Path(val_path).exists():
        val_dataset = PairwisePreferenceDataset(val_path, shuffle_pairs=False)
    
    # Create dataloaders
    collator = DataCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    
    return train_loader, val_loader

class EmbeddingPreprocessor:
    def __init__(
        self,
        model,
        cache_dir: str = None,
        batch_size: int = 32
    ):
        self.model = model
        if cache_dir is None:
            import os
            # Use cache directory relative to project root
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
    
    def precompute_dataset_embeddings(self, dataset_path: str):
        # Load dataset
        with open(dataset_path, 'r') as f:
            pairs = json.load(f)
        
        print(f"Precomputing embeddings for {len(pairs)} pairs...")
        
        all_texts = set()
        for pair in pairs:
            # Collect all unique texts
            all_texts.add(pair['task'])
            all_texts.add(pair['positive']['trajectory_full_context'])
            all_texts.add(pair['positive']['module_k_all_rounds'])
            all_texts.add(pair['negative']['trajectory_full_context'])
            all_texts.add(pair['negative']['module_k_all_rounds'])
        
        print(f"Total unique texts to encode: {len(all_texts)}")
        
        # Batch encode and cache
        texts_list = list(all_texts)
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts_list), self.batch_size):
                batch_texts = texts_list[i:i + self.batch_size]
                
                for text in batch_texts:
                    text_hash = self._compute_hash(text)
                    cache_path = self.cache_dir / f"{text_hash}.pt"
                    
                    if not cache_path.exists():
                        embedding = self.model.encode_text(text)
                        torch.save(embedding.cpu(), cache_path)
                
                if (i // self.batch_size) % 10 == 0:
                    print(f"  Processed {i}/{len(texts_list)} texts")
        
        print("Embedding precomputation complete!")
    
    def _compute_hash(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def load_embedding(self, text: str) -> Optional[torch.Tensor]:
        text_hash = self._compute_hash(text)
        cache_path = self.cache_dir / f"{text_hash}.pt"
        
        if cache_path.exists():
            return torch.load(cache_path)
        return None