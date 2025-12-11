"""
Low-Layer Memory Module

Stores recent raw steps in a FIFO buffer for short-term recall.
"""

from typing import List, Dict, Optional
from collections import deque
import torch


class LowLayerMemory:
    """
    Low-layer memory: FIFO buffer for recent steps
    
    Features:
    - Fine-grained: Preserves complete observation and action details
    - Temporal: Only keeps recent N steps
    - Efficient: O(1) append and retrieval
    """
    
    def __init__(
        self,
        max_size: int = 20,
        vision_hidden_size: int = 1536,
        store_on_cpu: bool = True
    ):
        """
        Args:
            max_size: Maximum number of steps to store
            vision_hidden_size: Dimension of vision features
            store_on_cpu: If True, store vision features on CPU to save GPU memory
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        
        self.max_size = max_size
        self.vision_hidden_size = vision_hidden_size
        self.store_on_cpu = store_on_cpu
        
        # Use deque for efficient FIFO operations
        self.steps = deque(maxlen=max_size)
        self.current_step_count = 0
    
    def add_step(
        self,
        vision_features: torch.Tensor,
        action_type: int,
        action_position: List[float],
        action_value: Optional[str] = None,
        reward: float = 0.0,
        timestamp: Optional[int] = None
    ) -> None:
        """
        Add a new step to the buffer
        
        Args:
            vision_features: (1, vision_hidden_size) or (vision_hidden_size,)
            action_type: Action type index
            action_position: List of coordinates [x, y] or [x1, y1, x2, y2]
            action_value: Optional text value for INPUT actions
            reward: Step reward
            timestamp: Optional explicit timestamp (uses current_step_count if None)
        """
        # Ensure vision_features is 2D
        if vision_features.dim() == 1:
            vision_features = vision_features.unsqueeze(0)
        
        # Move to CPU if requested
        if self.store_on_cpu and vision_features.is_cuda:
            vision_features = vision_features.cpu()
        
        step = {
            'vision_features': vision_features.detach(),  # Detach from computation graph
            'action_type': int(action_type),
            'action_position': list(action_position),
            'action_value': action_value,
            'reward': float(reward),
            'timestamp': timestamp if timestamp is not None else self.current_step_count,
            'step_idx': self.current_step_count
        }
        
        self.steps.append(step)
        self.current_step_count += 1
    
    def get_recent_steps(self, k: Optional[int] = None) -> List[Dict]:
        """
        Get the most recent K steps
        
        Args:
            k: Number of recent steps to retrieve (None returns all)
        
        Returns:
            List of step dictionaries
        """
        if k is None or k >= len(self.steps):
            return list(self.steps)
        
        if k <= 0:
            return []
        
        # Get last k steps from deque
        return list(self.steps)[-k:]
    
    def get_step_at(self, idx: int) -> Optional[Dict]:
        """
        Get step at specific index (relative to current buffer)
        
        Args:
            idx: Index in current buffer (0 to len-1)
        
        Returns:
            Step dict or None if index out of range
        """
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None
    
    def get_step_by_timestamp(self, timestamp: int) -> Optional[Dict]:
        """
        Get step by its timestamp
        
        Args:
            timestamp: Step timestamp
        
        Returns:
            Step dict or None if not found
        """
        for step in self.steps:
            if step['timestamp'] == timestamp:
                return step
        return None
    
    def clear(self):
        """Clear the buffer (e.g., at episode start)"""
        self.steps.clear()
        self.current_step_count = 0
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.steps)
    
    def __repr__(self):
        return f"LowLayerMemory(size={len(self)}/{self.max_size}, total_steps={self.current_step_count})"
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.steps:
            return {
                'size': 0,
                'max_size': self.max_size,
                'total_steps': self.current_step_count,
                'oldest_timestamp': None,
                'newest_timestamp': None
            }
        
        return {
            'size': len(self.steps),
            'max_size': self.max_size,
            'total_steps': self.current_step_count,
            'oldest_timestamp': self.steps[0]['timestamp'],
            'newest_timestamp': self.steps[-1]['timestamp'],
            'action_types': [step['action_type'] for step in self.steps]
        }

