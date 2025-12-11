"""
Trajectory Block Encoder

Encodes a trajectory block (sequence of steps) into a fixed-dimension vector
using self-attention mechanism and compression.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class TrajectoryBlockEncoder(nn.Module):
    """
    Encodes a trajectory block into a fixed-dimension embedding.
    
    Architecture:
    1. Step Projection: Project (vision_features, action_type, action_position) to unified space
    2. Positional Encoding: Add position encoding for step order within block
    3. Self-Attention: Transformer layers to aggregate information
    4. Compression: Query token or mean pooling to get fixed-size block embedding
    """
    
    def __init__(
        self,
        vision_hidden_size: int,
        action_vocab_size: int,
        position_dim: int = 4,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        compression_method: Literal['query', 'mean'] = 'query',
        max_block_length: int = 20
    ):
        """
        Args:
            vision_hidden_size: Dimension of vision encoder output (from model config)
            action_vocab_size: Number of distinct action types (from dataset)
            position_dim: Dimension of position vector (2 for [x,y], 4 for bbox)
            hidden_dim: Hidden dimension for embeddings and Transformer
            num_layers: Number of Transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            compression_method: 'query' (learnable query token) or 'mean' (average pooling)
            max_block_length: Maximum number of steps in a block
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.vision_hidden_size = vision_hidden_size
        self.action_vocab_size = action_vocab_size
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.compression_method = compression_method
        self.max_block_length = max_block_length
        
        # 1. Step Projection Layers
        self.vision_proj = nn.Linear(vision_hidden_size, hidden_dim)
        self.action_type_embed = nn.Embedding(action_vocab_size, hidden_dim)
        self.position_proj = nn.Linear(position_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional Encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_block_length, hidden_dim) * 0.02
        )
        
        # 3. Self-Attention Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 4. Compression Mechanism
        if compression_method == 'query':
            self.block_query = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * 0.02
            )
        elif compression_method != 'mean':
            raise ValueError(f"Unknown compression_method: {compression_method}. "
                           f"Choose from: 'query', 'mean'")
        
        # 5. Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode_step(
        self,
        vision_feat: torch.Tensor,
        action_type: torch.Tensor,
        action_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single step.
        
        Args:
            vision_feat: (B, vision_hidden_size)
            action_type: (B,) action type indices
            action_pos: (B, position_dim)
        
        Returns:
            step_emb: (B, hidden_dim)
        """
        v_emb = self.vision_proj(vision_feat)
        a_emb = self.action_type_embed(action_type)
        p_emb = self.position_proj(action_pos)
        
        concat = torch.cat([v_emb, a_emb, p_emb], dim=-1)
        step_emb = self.fusion(concat)
        
        return step_emb
    
    def forward(
        self,
        vision_features: torch.Tensor,
        action_types: torch.Tensor,
        action_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a batch of trajectory blocks.
        
        Args:
            vision_features: (B, num_steps, vision_hidden_size)
            action_types: (B, num_steps) action type indices
            action_positions: (B, num_steps, position_dim)
            attention_mask: (B, num_steps) 1=valid, 0=padding
        
        Returns:
            block_embeddings: (B, hidden_dim)
        """
        B, num_steps, _ = vision_features.shape
        
        if num_steps > self.max_block_length:
            raise ValueError(f"num_steps ({num_steps}) exceeds max_block_length ({self.max_block_length})")
        
        # 1. Encode each step
        step_embeddings = []
        for t in range(num_steps):
            step_emb = self.encode_step(
                vision_features[:, t, :],
                action_types[:, t],
                action_positions[:, t, :]
            )
            step_embeddings.append(step_emb)
        
        step_seq = torch.stack(step_embeddings, dim=1)
        
        # 2. Add positional encoding
        step_seq = step_seq + self.pos_encoding[:, :num_steps, :]
        
        # 3. Apply self-attention
        if self.compression_method == 'query':
            query = self.block_query.expand(B, -1, -1)
            seq_with_query = torch.cat([query, step_seq], dim=1)
            
            if attention_mask is not None:
                query_mask = torch.ones(B, 1, device=step_seq.device, dtype=attention_mask.dtype)
                full_mask = torch.cat([query_mask, attention_mask], dim=1)
                src_key_padding_mask = (full_mask == 0)
            else:
                src_key_padding_mask = None
            
            encoded = self.transformer(
                seq_with_query,
                src_key_padding_mask=src_key_padding_mask
            )
            
            block_emb = encoded[:, 0, :]
        
        elif self.compression_method == 'mean':
            if attention_mask is not None:
                src_key_padding_mask = (attention_mask == 0)
            else:
                src_key_padding_mask = None
            
            encoded = self.transformer(
                step_seq,
                src_key_padding_mask=src_key_padding_mask
            )
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                block_emb = (encoded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            else:
                block_emb = encoded.mean(dim=1)
        
        # 4. Output projection
        block_emb = self.output_proj(block_emb)
        
        return block_emb
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

