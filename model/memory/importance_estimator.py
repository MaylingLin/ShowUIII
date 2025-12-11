"""
Importance Estimator

Determines which steps are important enough to store in memory.
"""

from typing import List, Optional
import torch
import torch.nn.functional as F


class ImportanceEstimator:
    """
    Importance estimator: Decides whether a step is worth storing
    
    Evaluation criteria (all configurable):
    - Reward change (high/low reward steps)
    - Action diversity (new action types)
    - State change (visual feature difference)
    - Forced storage interval
    
    No hardcoded thresholds - all parameters configurable.
    """
    
    def __init__(
        self,
        reward_threshold: float = 0.5,
        negative_reward_threshold: float = -0.5,
        diversity_weight: float = 0.3,
        change_threshold: float = 0.2,
        force_store_interval: int = 10,
        importance_threshold: float = 0.5,
        recent_history_size: int = 20
    ):
        """
        Args:
            reward_threshold: Reward above this is considered important
            negative_reward_threshold: Reward below this is also important (failure)
            diversity_weight: Weight for action diversity importance
            change_threshold: Visual change threshold (1-cosine similarity)
            force_store_interval: Force storage every N steps
            importance_threshold: Overall importance score threshold for storage
            recent_history_size: Number of recent steps to keep in memory
        """
        if force_store_interval < 1:
            raise ValueError(f"force_store_interval must be >= 1, got {force_store_interval}")
        if recent_history_size < 1:
            raise ValueError(f"recent_history_size must be >= 1, got {recent_history_size}")
        
        self.reward_threshold = reward_threshold
        self.negative_reward_threshold = negative_reward_threshold
        self.diversity_weight = diversity_weight
        self.change_threshold = change_threshold
        self.force_store_interval = force_store_interval
        self.importance_threshold = importance_threshold
        self.recent_history_size = recent_history_size
        
        # History tracking
        self.recent_actions = []
        self.recent_vision_features = []
        self.step_count = 0
    
    def should_store(
        self,
        vision_features: torch.Tensor,
        action_type: int,
        reward: float
    ) -> bool:
        """
        Determine if step should be stored
        
        Args:
            vision_features: Vision features tensor
            action_type: Action type index
            reward: Step reward
        
        Returns:
            True if step should be stored
        """
        importance_score = 0.0
        
        # 1. Reward-based importance
        if reward > self.reward_threshold:
            importance_score += 0.4  # High reward
        elif reward < self.negative_reward_threshold:
            importance_score += 0.3  # Failure also important (negative example)
        
        # 2. Action diversity
        if action_type not in self.recent_actions[-5:]:
            importance_score += self.diversity_weight
        
        # 3. Visual change
        if self.recent_vision_features:
            last_vision = self.recent_vision_features[-1]
            
            # Ensure same shape
            vf_flat = vision_features.flatten()
            lv_flat = last_vision.flatten()
            
            if vf_flat.shape == lv_flat.shape:
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    vf_flat.unsqueeze(0),
                    lv_flat.unsqueeze(0),
                    dim=1
                ).item()
                
                # High change (low similarity) is important
                if similarity < (1 - self.change_threshold):
                    importance_score += 0.3
        
        # 4. Force storage at intervals
        # Check if next step_count will be at interval boundary
        if (self.step_count + 1) % self.force_store_interval == 0:
            importance_score += 0.5
        
        # Update history
        self.recent_actions.append(action_type)
        self.recent_vision_features.append(vision_features.detach().cpu())
        
        # Maintain history size
        if len(self.recent_actions) > self.recent_history_size:
            self.recent_actions.pop(0)
            self.recent_vision_features.pop(0)
        
        self.step_count += 1
        
        # Decision
        return importance_score > self.importance_threshold
    
    def reset(self):
        """Reset history (e.g., at episode start)"""
        self.recent_actions = []
        self.recent_vision_features = []
        self.step_count = 0
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        return {
            'step_count': self.step_count,
            'history_size': len(self.recent_actions),
            'unique_actions': len(set(self.recent_actions)),
            'config': {
                'reward_threshold': self.reward_threshold,
                'diversity_weight': self.diversity_weight,
                'change_threshold': self.change_threshold,
                'force_store_interval': self.force_store_interval,
                'importance_threshold': self.importance_threshold
            }
        }


class SelectiveMemoryUpdater:
    """
    Selective memory updater: Only stores important steps
    
    Wraps HierarchicalMemoryManager and filters steps through
    ImportanceEstimator before storage.
    """
    
    def __init__(
        self,
        memory_manager,  # HierarchicalMemoryManager instance
        importance_estimator: ImportanceEstimator
    ):
        """
        Args:
            memory_manager: HierarchicalMemoryManager to wrap
            importance_estimator: ImportanceEstimator for filtering
        """
        self.memory_manager = memory_manager
        self.importance_estimator = importance_estimator
        
        # Statistics
        self.skipped_count = 0
        self.stored_count = 0
    
    def update(
        self,
        vision_features: torch.Tensor,
        action_type: int,
        action_position: List[float],
        action_value: Optional[str],
        reward: float
    ) -> bool:
        """
        Selectively update memory
        
        Args:
            vision_features: Vision features
            action_type: Action type index
            action_position: Action coordinates
            action_value: Optional action value
            reward: Step reward
        
        Returns:
            True if step was stored, False if skipped
        """
        should_store = self.importance_estimator.should_store(
            vision_features,
            action_type,
            reward
        )
        
        if should_store:
            self.memory_manager.on_step(
                vision_features,
                action_type,
                action_position,
                action_value,
                reward
            )
            self.stored_count += 1
            return True
        else:
            self.skipped_count += 1
            return False
    
    def reset(self):
        """Reset statistics"""
        self.importance_estimator.reset()
        self.skipped_count = 0
        self.stored_count = 0
    
    def get_stats(self) -> dict:
        """Get update statistics"""
        total = self.stored_count + self.skipped_count
        return {
            'stored': self.stored_count,
            'skipped': self.skipped_count,
            'total': total,
            'storage_rate': self.stored_count / total if total > 0 else 0.0,
            'estimator_stats': self.importance_estimator.get_stats()
        }
