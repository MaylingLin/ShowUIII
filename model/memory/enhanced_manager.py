"""
Enhanced Hierarchical Memory Manager

Integrates dynamic update and adaptive retrieval mechanisms.
"""

import torch
from typing import List, Dict, Tuple, Optional
import logging

from .memory_manager import HierarchicalMemoryManager
from .query_classifier import HeuristicQueryClassifier, LearnableQueryClassifier
from .importance_estimator import ImportanceEstimator, SelectiveMemoryUpdater


logger = logging.getLogger(__name__)


class EnhancedHierarchicalMemoryManager(HierarchicalMemoryManager):
    """
    Enhanced memory manager with dynamic update and adaptive retrieval
    
    Features:
    - Query-aware retrieval (adjusts layer weights by query type)
    - Selective storage (only stores important steps)
    - Adaptive layer weighting
    
    All features are optional and configurable.
    """
    
    def __init__(
        self,
        *args,
        # Adaptive retrieval settings
        use_query_classifier: bool = True,
        query_classifier_type: str = 'heuristic',  # 'heuristic' or 'learnable'
        query_classifier_config: Optional[Dict] = None,
        
        # Selective storage settings
        use_selective_storage: bool = True,
        importance_estimator_config: Optional[Dict] = None,
        
        **kwargs
    ):
        """
        Args:
            use_query_classifier: Enable query-aware retrieval
            query_classifier_type: 'heuristic' (no training) or 'learnable'
            query_classifier_config: Config dict for query classifier
            use_selective_storage: Enable selective step storage
            importance_estimator_config: Config dict for importance estimator
            *args, **kwargs: Passed to HierarchicalMemoryManager
        """
        super().__init__(*args, **kwargs)
        
        # Query classifier
        self.use_query_classifier = use_query_classifier
        if use_query_classifier:
            if query_classifier_type == 'heuristic':
                config = query_classifier_config or {}
                self.query_classifier = HeuristicQueryClassifier(**config)
                logger.info("Initialized HeuristicQueryClassifier")
            elif query_classifier_type == 'learnable':
                config = query_classifier_config or {'query_dim': self.block_encoder.vision_hidden_size}
                self.query_classifier = LearnableQueryClassifier(**config)
                logger.info("Initialized LearnableQueryClassifier")
            else:
                raise ValueError(f"Unknown query_classifier_type: {query_classifier_type}")
        else:
            self.query_classifier = None
        
        # Selective storage
        self.use_selective_storage = use_selective_storage
        if use_selective_storage:
            config = importance_estimator_config or {}
            importance_estimator = ImportanceEstimator(**config)
            self.selective_updater = SelectiveMemoryUpdater(self, importance_estimator)
            logger.info("Initialized SelectiveMemoryUpdater")
        else:
            self.selective_updater = None
        
        # Track recent rewards for query classification
        self.recent_rewards = []
        self.max_recent_rewards = 10
    
    def on_episode_start(
        self,
        task_description: str,
        task_embedding: torch.Tensor
    ):
        """Override to reset selective updater"""
        super().on_episode_start(task_description, task_embedding)
        
        # Reset recent rewards
        self.recent_rewards = []
        
        # Reset selective updater
        if self.selective_updater:
            self.selective_updater.reset()
    
    def on_step_adaptive(
        self,
        vision_features: torch.Tensor,
        action_type: int,
        action_position: List[float],
        action_value: Optional[str] = None,
        reward: float = 0.0
    ) -> bool:
        """
        Adaptive step update with selective storage
        
        Args:
            vision_features: Vision features
            action_type: Action type index
            action_position: Action coordinates
            action_value: Optional action value
            reward: Step reward
        
        Returns:
            True if step was stored, False if skipped
        """
        # Update recent rewards
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.max_recent_rewards:
            self.recent_rewards.pop(0)
        
        # Use selective storage if enabled
        if self.use_selective_storage and self.selective_updater:
            stored = self.selective_updater.update(
                vision_features,
                action_type,
                action_position,
                action_value,
                reward
            )
            return stored
        else:
            # Default: store all steps
            self.on_step(
                vision_features,
                action_type,
                action_position,
                action_value,
                reward
            )
            return True
    
    def retrieve_memory_context_adaptive(
        self,
        current_obs: torch.Tensor,
        task_progress: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Adaptive retrieval with query-aware layer weighting
        
        Args:
            current_obs: (B, query_dim) current observation
            task_progress: Optional task progress (0-1)
        
        Returns:
            memory_context: (B, output_dim) fused memory context
            retrieval_info: Dict with retrieval details
        """
        # Compute task progress if not provided
        if task_progress is None:
            total_subgoals = len(self.high_memory.subgoals)
            if total_subgoals > 0:
                completed = sum(1 for sg in self.high_memory.subgoals if sg['status'] == 'completed')
                task_progress = completed / total_subgoals
            else:
                task_progress = self.current_step_count / 100.0  # Rough estimate
                task_progress = min(task_progress, 1.0)
        
        # Use query classifier if enabled
        if self.use_query_classifier and self.query_classifier:
            if isinstance(self.query_classifier, HeuristicQueryClassifier):
                # Heuristic classifier
                query_weights = self.query_classifier.classify(
                    self.current_step_count,
                    self.recent_rewards,
                    task_progress
                )
                # Convert to tensor
                layer_weights_np = [
                    query_weights['detail'],
                    query_weights['planning'],
                    query_weights['global']
                ]
                retrieval_method = 'heuristic'
                
            else:
                # Learnable classifier
                with torch.no_grad():
                    classifier_output = self.query_classifier(current_obs)
                    layer_weights_np = classifier_output['query_type_probs'][0].cpu().numpy().tolist()
                retrieval_method = 'learnable'
            
            # Standard retrieval with custom weights
            memory_context, info = self.retrieve_memory_context(current_obs)
            
            # Add query classifier info
            info['query_weights'] = layer_weights_np
            info['retrieval_method'] = retrieval_method
        else:
            # Standard retrieval
            memory_context, info = self.retrieve_memory_context(current_obs)
            info['retrieval_method'] = 'standard'
        
        return memory_context, info
    
    def get_enhanced_stats(self) -> Dict:
        """Get comprehensive statistics including adaptive features"""
        base_stats = self.get_memory_stats()
        
        enhanced_stats = {
            **base_stats,
            'adaptive_features': {
                'query_classifier_enabled': self.use_query_classifier,
                'selective_storage_enabled': self.use_selective_storage
            }
        }
        
        # Add selective storage stats
        if self.selective_updater:
            enhanced_stats['selective_storage'] = self.selective_updater.get_stats()
        
        # Add query classifier config
        if self.query_classifier and isinstance(self.query_classifier, HeuristicQueryClassifier):
            enhanced_stats['query_classifier_config'] = self.query_classifier.get_config()
        
        return enhanced_stats
    
    def __repr__(self):
        base_repr = super().__repr__()
        features = []
        if self.use_query_classifier:
            features.append("query_aware")
        if self.use_selective_storage:
            features.append("selective")
        
        if features:
            return f"Enhanced{base_repr[:-1]}, features={features})"
        return base_repr
