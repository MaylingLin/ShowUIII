"""
Action Vocabulary Builder

Dynamically builds action vocabulary from dataset to avoid hardcoding.
"""

import json
import os
from typing import List, Dict, Set
from collections import Counter


class ActionVocabulary:
    """Manages action type vocabulary extracted from dataset"""
    
    def __init__(self):
        self.action_to_idx = {}
        self.idx_to_action = {}
        self.action_counts = Counter()
    
    def build_from_dataset(
        self,
        dataset_dir: str,
        dataset_name: str,
        json_file: str
    ) -> None:
        """
        Build vocabulary from dataset JSON file.
        
        Args:
            dataset_dir: Base dataset directory
            dataset_name: Dataset name (e.g., 'GUI_Course/GUIAct', 'Mind2Web')
            json_file: JSON metadata file name
        """
        # Construct path
        meta_path = os.path.join(dataset_dir, dataset_name, "metadata", f"{json_file}.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
        
        # Load and scan for action types
        with open(meta_path, 'r') as f:
            data = json.load(f)
        
        action_set = set()
        
        for sample in data:
            # Handle different data formats
            if 'actions' in sample:
                # Mind2Web / GUIAct format
                for step in sample['actions']:
                    action_type = self._extract_action_type(step)
                    if action_type:
                        action_set.add(action_type.upper())
                        self.action_counts[action_type.upper()] += 1
            
            elif 'action_reprs' in sample:
                # Alternative format
                for action_repr in sample['action_reprs']:
                    if isinstance(action_repr, dict) and 'action' in action_repr:
                        action_type = action_repr['action']
                        if action_type:
                            action_set.add(action_type.upper())
                            self.action_counts[action_type.upper()] += 1
        
        # Build vocabulary (sorted for consistency)
        sorted_actions = sorted(list(action_set))
        
        # Reserve index 0 for padding
        self.action_to_idx = {'<PAD>': 0}
        self.idx_to_action = {0: '<PAD>'}
        
        for idx, action in enumerate(sorted_actions, start=1):
            self.action_to_idx[action] = idx
            self.idx_to_action[idx] = action
    
    def _extract_action_type(self, step: dict) -> str:
        """Extract action type from step dict"""
        if 'action_type' in step:
            return step['action_type']
        elif 'action' in step:
            if isinstance(step['action'], dict):
                return step['action'].get('action', '')
            else:
                return step['action']
        return ''
    
    def encode(self, action: str) -> int:
        """Convert action string to index"""
        action_upper = action.upper().strip()
        return self.action_to_idx.get(action_upper, 0)  # Return PAD if unknown
    
    def decode(self, idx: int) -> str:
        """Convert index to action string"""
        return self.idx_to_action.get(idx, '<PAD>')
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.action_to_idx)
    
    def get_stats(self) -> Dict:
        """Get vocabulary statistics"""
        return {
            'vocab_size': len(self),
            'actions': list(self.action_to_idx.keys()),
            'counts': dict(self.action_counts)
        }
    
    def save(self, path: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'action_to_idx': self.action_to_idx,
            'idx_to_action': {int(k): v for k, v in self.idx_to_action.items()},
            'action_counts': dict(self.action_counts)
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.action_to_idx = vocab_data['action_to_idx']
        self.idx_to_action = {int(k): v for k, v in vocab_data['idx_to_action'].items()}
        self.action_counts = Counter(vocab_data['action_counts'])


def get_action_vocab_for_dataset(
    dataset_dir: str,
    dataset_name: str,
    json_file: str,
    cache_dir: str = None
) -> ActionVocabulary:
    """
    Get action vocabulary for a dataset, using cache if available.
    
    Args:
        dataset_dir: Base dataset directory
        dataset_name: Dataset name
        json_file: JSON file name
        cache_dir: Directory to cache vocabulary files
    
    Returns:
        ActionVocabulary instance
    """
    vocab = ActionVocabulary()
    
    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"action_vocab_{dataset_name.replace('/', '_')}_{json_file}.json"
        )
        
        if os.path.exists(cache_file):
            print(f"Loading cached action vocabulary from {cache_file}")
            vocab.load(cache_file)
            return vocab
    
    # Build from dataset
    print(f"Building action vocabulary from {dataset_name}/{json_file}")
    vocab.build_from_dataset(dataset_dir, dataset_name, json_file)
    
    # Save to cache
    if cache_dir:
        vocab.save(cache_file)
        print(f"Saved action vocabulary to {cache_file}")
    
    return vocab

