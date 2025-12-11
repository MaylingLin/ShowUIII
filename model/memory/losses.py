"""
Loss Functions for Block Encoder Training

Implements auxiliary losses to train the block encoder to preserve
semantically meaningful information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BlockEncoderLoss(nn.Module):
    """
    Combined loss for training trajectory block encoder.
    
    Includes:
    1. Action Presence Loss: Predict which actions occurred in the block
    2. Optional Contrastive Loss: Blocks from same task should be similar
    """
    
    def __init__(
        self,
        action_vocab_size: int,
        hidden_dim: int,
        use_contrastive: bool = False,
        contrastive_temperature: float = 0.07,
        action_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 0.1
    ):
        """
        Args:
            action_vocab_size: Number of action types
            hidden_dim: Dimension of block embeddings
            use_contrastive: Whether to use contrastive learning
            contrastive_temperature: Temperature for contrastive loss
            action_loss_weight: Weight for action presence loss
            contrastive_loss_weight: Weight for contrastive loss
        """
        super().__init__()
        
        self.action_vocab_size = action_vocab_size
        self.hidden_dim = hidden_dim
        self.use_contrastive = use_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.action_loss_weight = action_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        
        # Action presence prediction head
        self.action_presence_head = nn.Linear(hidden_dim, action_vocab_size)
        
        # Initialize
        nn.init.xavier_uniform_(self.action_presence_head.weight)
        nn.init.zeros_(self.action_presence_head.bias)
    
    def compute_action_presence_loss(
        self,
        block_embeddings: torch.Tensor,
        block_action_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for predicting which actions occurred in the block.
        
        Args:
            block_embeddings: (B, hidden_dim) block representations
            block_action_types: (B, num_steps) action type indices in the block
            attention_mask: (B, num_steps) mask for valid actions
        
        Returns:
            loss: scalar tensor
        """
        B = block_embeddings.shape[0]
        
        # Predict action presence (multi-label classification)
        action_logits = self.action_presence_head(block_embeddings)  # (B, vocab_size)
        
        # Create target: 1 if action occurred in block, 0 otherwise
        action_labels = torch.zeros_like(action_logits)
        
        for b in range(B):
            if attention_mask is not None:
                valid_actions = block_action_types[b][attention_mask[b] == 1]
            else:
                valid_actions = block_action_types[b]
            
            # Mark occurred actions as 1
            unique_actions = torch.unique(valid_actions)
            # Filter out padding index if exists (typically 0 or -100)
            unique_actions = unique_actions[unique_actions >= 0]
            if len(unique_actions) > 0:
                action_labels[b, unique_actions] = 1.0
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(action_logits, action_labels)
        
        return loss
    
    def compute_contrastive_loss(
        self,
        block_embeddings: torch.Tensor,
        task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss: blocks from same task should be similar.
        
        Args:
            block_embeddings: (B, hidden_dim) block representations
            task_ids: (B,) task identifier for each block
        
        Returns:
            loss: scalar tensor
        """
        # Normalize embeddings
        block_embeddings_norm = F.normalize(block_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(block_embeddings_norm, block_embeddings_norm.T)
        sim_matrix = sim_matrix / self.contrastive_temperature
        
        # Create labels: 1 if same task, 0 otherwise
        task_ids = task_ids.unsqueeze(0)  # (1, B)
        labels = (task_ids == task_ids.T).float()  # (B, B)
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(labels.shape[0], device=labels.device).bool()
        labels = labels.masked_fill(mask, 0)
        
        # Compute InfoNCE-style loss
        # Positive pairs should have high similarity, negative pairs low
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(mask, 0)  # Exclude self
        
        # For each block, compute loss
        pos_sim = (exp_sim * labels).sum(dim=1)  # Sum of positive pairs
        all_sim = exp_sim.sum(dim=1)  # Sum of all pairs
        
        # Avoid division by zero
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))
        loss = loss.mean()
        
        return loss
    
    def forward(
        self,
        block_embeddings: torch.Tensor,
        block_action_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            block_embeddings: (B, hidden_dim)
            block_action_types: (B, num_steps)
            attention_mask: (B, num_steps) optional
            task_ids: (B,) optional, for contrastive loss
        
        Returns:
            dict with keys:
                - 'total_loss': combined loss
                - 'action_loss': action presence loss
                - 'contrastive_loss': contrastive loss (if enabled)
        """
        losses = {}
        
        # Action presence loss
        action_loss = self.compute_action_presence_loss(
            block_embeddings,
            block_action_types,
            attention_mask
        )
        losses['action_loss'] = action_loss
        
        # Total loss starts with action loss
        total_loss = self.action_loss_weight * action_loss
        
        # Contrastive loss (optional)
        if self.use_contrastive and task_ids is not None:
            contrastive_loss = self.compute_contrastive_loss(
                block_embeddings,
                task_ids
            )
            losses['contrastive_loss'] = contrastive_loss
            total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss
        
        losses['total_loss'] = total_loss
        
        return losses

