"""
Cross-Layer Retrieval Module

Retrieves and fuses information from three memory layers using attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class CrossLayerRetrieval(nn.Module):
    """
    Cross-layer retrieval: Retrieve and fuse information from three memory layers
    
    Features:
    - Differentiable: Supports end-to-end training
    - Adaptive: Dynamically weights layer importance
    - Efficient: Parallel retrieval from all layers
    """
    
    def __init__(
        self,
        query_dim: int,
        low_layer_dim: int,
        mid_layer_dim: int,
        high_layer_dim: int,
        output_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of current observation (from vision encoder)
            low_layer_dim: Dimension of low-layer step features
            mid_layer_dim: Dimension of mid-layer block embeddings
            high_layer_dim: Dimension of high-layer task embeddings
            output_dim: Dimension of fused memory context
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        if output_dim % num_heads != 0:
            raise ValueError(f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})")
        
        self.query_dim = query_dim
        self.low_layer_dim = low_layer_dim
        self.mid_layer_dim = mid_layer_dim
        self.high_layer_dim = high_layer_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Query encoder: Encodes current observation to query vector
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Projection layers: Project each layer to unified space
        self.low_proj = nn.Linear(low_layer_dim, output_dim)
        self.mid_proj = nn.Linear(mid_layer_dim, output_dim)
        self.high_proj = nn.Linear(high_layer_dim, output_dim)
        
        # Multi-head attention for retrieval
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer weight network: Learn importance of each layer
        self.layer_weight_net = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
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
        current_obs: torch.Tensor,
        low_memory: List[Dict],
        mid_memory: List[torch.Tensor],
        high_memory: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Retrieve and fuse memory from three layers
        
        Args:
            current_obs: (B, query_dim) current observation
            low_memory: List of step dicts from LowLayerMemory
            mid_memory: List of block embeddings from MidLayerMemory
            high_memory: Task context dict from HighLayerMemory
        
        Returns:
            memory_context: (B, output_dim) fused memory context
            retrieval_info: Dict with retrieval details
        """
        B = current_obs.shape[0]
        device = current_obs.device
        
        # 1. Encode query
        query = self.query_encoder(current_obs)  # (B, output_dim)
        query_expanded = query.unsqueeze(1)  # (B, 1, output_dim)
        
        # 2. Retrieve from each layer
        low_context, low_attn = self._retrieve_from_low_layer(query_expanded, low_memory, device)
        mid_context, mid_attn = self._retrieve_from_mid_layer(query_expanded, mid_memory, device)
        high_context, high_attn = self._retrieve_from_high_layer(query_expanded, high_memory, device)
        
        # 3. Learn layer weights
        layer_weights = self.layer_weight_net(query)  # (B, 3)
        
        # 4. Weighted fusion
        fused_context = (
            layer_weights[:, 0:1] * low_context +
            layer_weights[:, 1:2] * mid_context +
            layer_weights[:, 2:3] * high_context
        )
        
        # 5. Final fusion
        memory_context = self.fusion(fused_context)
        
        # 6. Collect retrieval info
        retrieval_info = {
            'layer_weights': layer_weights.detach(),
            'low_context': low_context.detach(),
            'mid_context': mid_context.detach(),
            'high_context': high_context.detach(),
            'low_attention': low_attn,
            'mid_attention': mid_attn,
            'high_attention': high_attn
        }
        
        return memory_context, retrieval_info
    
    def _retrieve_from_low_layer(
        self,
        query: torch.Tensor,
        low_memory: List[Dict],
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from low-layer memory
        
        Returns:
            context: (B, output_dim)
            attention_weights: (B, 1, num_steps) or None
        """
        B = query.shape[0]
        
        if not low_memory:
            return torch.zeros(B, self.output_dim, device=device), None
        
        # Extract and stack vision features
        try:
            low_features = torch.stack([
                step['vision_features'].squeeze(0) for step in low_memory
            ], dim=0).to(device)  # (num_steps, low_dim)
        except Exception:
            return torch.zeros(B, self.output_dim, device=device), None
        
        # Project to output_dim
        low_features = self.low_proj(low_features).unsqueeze(0)  # (1, num_steps, output_dim)
        low_features = low_features.expand(B, -1, -1)  # (B, num_steps, output_dim)
        
        # Attention retrieval
        attn_output, attn_weights = self.attention(query, low_features, low_features)
        
        return attn_output.squeeze(1), attn_weights  # (B, output_dim), (B, 1, num_steps)
    
    def _retrieve_from_mid_layer(
        self,
        query: torch.Tensor,
        mid_memory: List[torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from mid-layer memory
        
        Returns:
            context: (B, output_dim)
            attention_weights: (B, 1, num_blocks) or None
        """
        B = query.shape[0]
        
        if not mid_memory:
            return torch.zeros(B, self.output_dim, device=device), None
        
        # Stack block embeddings
        try:
            mid_features = torch.cat(mid_memory, dim=0).to(device)  # (num_blocks, mid_dim)
        except Exception:
            return torch.zeros(B, self.output_dim, device=device), None
        
        # Project
        mid_features = self.mid_proj(mid_features).unsqueeze(0)  # (1, num_blocks, output_dim)
        mid_features = mid_features.expand(B, -1, -1)  # (B, num_blocks, output_dim)
        
        # Attention retrieval
        attn_output, attn_weights = self.attention(query, mid_features, mid_features)
        
        return attn_output.squeeze(1), attn_weights
    
    def _retrieve_from_high_layer(
        self,
        query: torch.Tensor,
        high_memory: Dict,
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from high-layer memory
        
        Returns:
            context: (B, output_dim)
            attention_weights: (B, 1, 1) or None
        """
        B = query.shape[0]
        
        # Check if task intent exists
        if high_memory.get('task_intent') is None:
            return torch.zeros(B, self.output_dim, device=device), None
        
        # Get task intent
        task_intent = high_memory['task_intent'].to(device)  # (1, high_dim)
        
        # Project
        task_intent = self.high_proj(task_intent).unsqueeze(0)  # (1, 1, output_dim)
        task_intent = task_intent.expand(B, -1, -1)  # (B, 1, output_dim)
        
        # Attention retrieval
        attn_output, attn_weights = self.attention(query, task_intent, task_intent)
        
        return attn_output.squeeze(1), attn_weights
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

