"""
Query Type Classifier

Identifies query type to determine which memory layer should be prioritized.
"""

from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeuristicQueryClassifier:
    """
    Heuristic query classifier (no training required)
    
    Classifies queries based on task state features:
    - detail: Low-layer memory (recent step details)
    - planning: Mid-layer memory (trajectory blocks, subgoals)
    - global: High-layer memory (task intent)
    
    All thresholds are configurable to avoid hardcoding.
    """
    
    def __init__(
        self,
        early_progress_threshold: float = 0.2,
        late_progress_threshold: float = 0.8,
        high_reward_threshold: float = 0.5,
        low_reward_threshold: float = 0.2,
        early_global_boost: float = 0.3,
        late_detail_boost: float = 0.2,
        mid_planning_boost: float = 0.2,
        success_detail_boost: float = 0.1,
        failure_planning_boost: float = 0.15
    ):
        """
        Args:
            early_progress_threshold: Task progress below this is considered "early"
            late_progress_threshold: Task progress above this is considered "late"
            high_reward_threshold: Reward above this is considered "success"
            low_reward_threshold: Reward below this is considered "failure"
            early_global_boost: Weight boost for global layer in early stage
            late_detail_boost: Weight boost for detail layer in late stage
            mid_planning_boost: Weight boost for planning layer in mid stage
            success_detail_boost: Weight boost for detail when succeeding
            failure_planning_boost: Weight boost for planning when failing
        """
        self.early_threshold = early_progress_threshold
        self.late_threshold = late_progress_threshold
        self.high_reward_threshold = high_reward_threshold
        self.low_reward_threshold = low_reward_threshold
        
        # Weight adjustment parameters (all configurable)
        self.early_global_boost = early_global_boost
        self.late_detail_boost = late_detail_boost
        self.mid_planning_boost = mid_planning_boost
        self.success_detail_boost = success_detail_boost
        self.failure_planning_boost = failure_planning_boost
        
        self.query_types = ['detail', 'planning', 'global']
    
    def classify(
        self,
        current_step: int,
        recent_rewards: List[float],
        task_progress: float
    ) -> Dict[str, float]:
        """
        Classify query based on heuristic rules
        
        Args:
            current_step: Current step number
            recent_rewards: List of recent rewards
            task_progress: Task progress (0-1)
        
        Returns:
            Dict with layer weights: {'detail': float, 'planning': float, 'global': float}
            Weights sum to 1.0
        """
        # Initialize with uniform weights
        weights = [1.0/3, 1.0/3, 1.0/3]  # [detail, planning, global]
        
        # Task stage adjustment
        if task_progress < self.early_threshold:
            # Early stage: rely more on global intent
            weights[2] += self.early_global_boost
            weights[0] -= self.early_global_boost / 2
            weights[1] -= self.early_global_boost / 2
            
        elif task_progress > self.late_threshold:
            # Late stage: rely more on details (finishing up)
            weights[0] += self.late_detail_boost
            weights[2] -= self.late_detail_boost
            
        else:
            # Mid stage: rely more on planning
            weights[1] += self.mid_planning_boost
            weights[2] -= self.mid_planning_boost
        
        # Recent performance adjustment
        if recent_rewards and len(recent_rewards) >= 3:
            avg_recent_reward = sum(recent_rewards[-3:]) / 3
            
            if avg_recent_reward > self.high_reward_threshold:
                # Success: learn from details
                weights[0] += self.success_detail_boost
                weights[1] -= self.success_detail_boost / 2
                weights[2] -= self.success_detail_boost / 2
                
            elif avg_recent_reward < self.low_reward_threshold:
                # Failure: re-plan
                weights[1] += self.failure_planning_boost
                weights[0] -= self.failure_planning_boost / 2
                weights[2] -= self.failure_planning_boost / 2
        
        # Normalize to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0/3, 1.0/3, 1.0/3]
        
        return {
            'detail': weights[0],
            'planning': weights[1],
            'global': weights[2]
        }
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            'early_threshold': self.early_threshold,
            'late_threshold': self.late_threshold,
            'high_reward_threshold': self.high_reward_threshold,
            'low_reward_threshold': self.low_reward_threshold,
            'early_global_boost': self.early_global_boost,
            'late_detail_boost': self.late_detail_boost,
            'mid_planning_boost': self.mid_planning_boost,
            'success_detail_boost': self.success_detail_boost,
            'failure_planning_boost': self.failure_planning_boost
        }


class LearnableQueryClassifier(nn.Module):
    """
    Learnable query classifier (requires training)
    
    Optional trainable version that learns query type classification
    from experience. All architecture dimensions are configurable.
    """
    
    def __init__(
        self,
        query_dim: int,
        hidden_dim: int = 256,
        num_query_types: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query features (from vision encoder)
            hidden_dim: Hidden layer dimension
            num_query_types: Number of query types (3: detail/planning/global)
            dropout: Dropout probability
        """
        super().__init__()
        
        if query_dim < 1:
            raise ValueError(f"query_dim must be >= 1, got {query_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.num_query_types = num_query_types
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Type classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_query_types)
        )
        
        # Learnable query type prototypes (for interpretability)
        self.query_prototypes = nn.Parameter(
            torch.randn(num_query_types, hidden_dim) * 0.02
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query_features: torch.Tensor  # (B, query_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Classify query type
        
        Returns:
            Dict with:
                - 'query_type_logits': (B, num_query_types)
                - 'query_type_probs': (B, num_query_types)
                - 'query_type': (B,) predicted type index
                - 'prototype_similarities': (B, num_query_types)
        """
        # Encode query
        query_encoded = self.query_encoder(query_features)  # (B, hidden_dim)
        
        # Classify
        logits = self.classifier(query_encoded)  # (B, num_query_types)
        probs = F.softmax(logits, dim=-1)
        predicted_type = torch.argmax(probs, dim=-1)
        
        # Compute similarity to prototypes (for interpretability)
        query_norm = F.normalize(query_encoded, p=2, dim=-1)
        proto_norm = F.normalize(self.query_prototypes, p=2, dim=-1)
        similarities = torch.matmul(query_norm, proto_norm.T)  # (B, num_query_types)
        
        return {
            'query_type_logits': logits,
            'query_type_probs': probs,
            'query_type': predicted_type,
            'prototype_similarities': similarities
        }
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

