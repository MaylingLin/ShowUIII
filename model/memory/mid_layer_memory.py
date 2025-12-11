"""
Mid-Layer Memory Module

Stores compressed trajectory blocks with semantic indexing for efficient retrieval.
"""

from typing import List, Dict, Tuple, Optional, Literal
import torch
import torch.nn.functional as F


class MidLayerMemory:
    """
    Mid-layer memory: Stores trajectory block embeddings
    
    Features:
    - Semantic compression: Uses TrajectoryBlockEncoder outputs
    - Efficient retrieval: Based on embedding similarity
    - Rich metadata: Preserves block action summaries
    """
    
    def __init__(
        self,
        block_hidden_dim: int,
        max_blocks: int = 100,
        similarity_metric: Literal['cosine', 'dot'] = 'cosine',
        store_on_cpu: bool = True
    ):
        """
        Args:
            block_hidden_dim: Dimension of block embeddings (from TrajectoryBlockEncoder)
            max_blocks: Maximum number of blocks to store
            similarity_metric: 'cosine' or 'dot' for similarity computation
            store_on_cpu: If True, store embeddings on CPU to save GPU memory
        """
        if block_hidden_dim < 1:
            raise ValueError(f"block_hidden_dim must be >= 1, got {block_hidden_dim}")
        if max_blocks < 1:
            raise ValueError(f"max_blocks must be >= 1, got {max_blocks}")
        if similarity_metric not in ['cosine', 'dot']:
            raise ValueError(f"similarity_metric must be 'cosine' or 'dot', got {similarity_metric}")
        
        self.block_hidden_dim = block_hidden_dim
        self.max_blocks = max_blocks
        self.similarity_metric = similarity_metric
        self.store_on_cpu = store_on_cpu
        
        # Storage
        self.block_embeddings = []  # List of (1, block_hidden_dim) tensors
        self.block_metadata = []    # List of metadata dicts
        self.block_count = 0  # Total blocks created (including removed ones)
    
    def add_block(
        self,
        block_embedding: torch.Tensor,
        block_metadata: Dict
    ) -> int:
        """
        Add a new block to the memory pool
        
        Args:
            block_embedding: (1, block_hidden_dim) from TrajectoryBlockEncoder
            block_metadata: Dict with keys:
                - 'actions': List[int] action type indices
                - 'positions': List[List[float]] action positions
                - 'rewards': List[float] step rewards
                - 'start_step': int
                - 'end_step': int
                - 'timestamp': int
        
        Returns:
            block_id: Index of the newly added block
        """
        # Validate input
        if block_embedding.dim() != 2 or block_embedding.shape[1] != self.block_hidden_dim:
            raise ValueError(
                f"block_embedding must have shape (1, {self.block_hidden_dim}), "
                f"got {block_embedding.shape}"
            )
        
        # Move to CPU if requested
        if self.store_on_cpu and block_embedding.is_cuda:
            block_embedding = block_embedding.cpu()
        
        # Store
        self.block_embeddings.append(block_embedding.detach())
        self.block_metadata.append(block_metadata)
        
        # FIFO: Remove oldest if exceeds capacity
        if len(self.block_embeddings) > self.max_blocks:
            self.block_embeddings.pop(0)
            self.block_metadata.pop(0)
        
        block_id = self.block_count
        self.block_count += 1
        
        return block_id
    
    def retrieve_similar_blocks(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve blocks similar to the query
        
        Args:
            query_embedding: (1, block_hidden_dim) query vector
            top_k: Number of top similar blocks to return
            return_scores: If True, return similarity scores
        
        Returns:
            List of (block_id, similarity_score, metadata) tuples
            Sorted by similarity (highest first)
        """
        if not self.block_embeddings:
            return []
        
        # Stack all block embeddings
        all_embeddings = torch.cat(self.block_embeddings, dim=0)  # (num_blocks, block_hidden_dim)
        query = query_embedding.to(all_embeddings.device)
        
        # Compute similarities
        if self.similarity_metric == 'cosine':
            query_norm = F.normalize(query, p=2, dim=-1)
            emb_norm = F.normalize(all_embeddings, p=2, dim=-1)
            similarities = torch.matmul(query_norm, emb_norm.T).squeeze(0)  # (num_blocks,)
        elif self.similarity_metric == 'dot':
            similarities = torch.matmul(query, all_embeddings.T).squeeze(0)
        
        # Top-K
        actual_top_k = min(top_k, len(self.block_embeddings))
        top_scores, top_indices = torch.topk(similarities, actual_top_k)
        
        # Build results
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            # Calculate absolute block_id (accounting for FIFO removal)
            absolute_block_id = self.block_count - len(self.block_embeddings) + idx
            
            if return_scores:
                results.append((absolute_block_id, score, self.block_metadata[idx]))
            else:
                results.append((absolute_block_id, self.block_metadata[idx]))
        
        return results
    
    def get_recent_blocks(self, k: int = 5) -> List[Tuple[torch.Tensor, Dict]]:
        """
        Get the most recent K blocks
        
        Args:
            k: Number of recent blocks to retrieve
        
        Returns:
            List of (embedding, metadata) tuples
        """
        k = min(k, len(self.block_embeddings))
        
        if k <= 0:
            return []
        
        return list(zip(self.block_embeddings[-k:], self.block_metadata[-k:]))
    
    def get_block_by_id(self, block_id: int) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Get block by its absolute ID
        
        Args:
            block_id: Absolute block ID
        
        Returns:
            (embedding, metadata) tuple or None if not found
        """
        # Calculate relative index
        oldest_id = self.block_count - len(self.block_embeddings)
        
        if block_id < oldest_id or block_id >= self.block_count:
            return None
        
        relative_idx = block_id - oldest_id
        return (self.block_embeddings[relative_idx], self.block_metadata[relative_idx])
    
    def clear(self):
        """Clear all blocks (e.g., at episode start)"""
        self.block_embeddings = []
        self.block_metadata = []
        self.block_count = 0
    
    def __len__(self):
        """Return current number of blocks"""
        return len(self.block_embeddings)
    
    def __repr__(self):
        return f"MidLayerMemory(size={len(self)}/{self.max_blocks}, total_blocks={self.block_count})"
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.block_embeddings:
            return {
                'size': 0,
                'max_blocks': self.max_blocks,
                'total_blocks': self.block_count,
                'oldest_block_id': None,
                'newest_block_id': None
            }
        
        oldest_id = self.block_count - len(self.block_embeddings)
        
        # Calculate average block size (number of actions)
        block_sizes = [len(meta.get('actions', [])) for meta in self.block_metadata]
        avg_block_size = sum(block_sizes) / len(block_sizes) if block_sizes else 0
        
        return {
            'size': len(self.block_embeddings),
            'max_blocks': self.max_blocks,
            'total_blocks': self.block_count,
            'oldest_block_id': oldest_id,
            'newest_block_id': self.block_count - 1,
            'avg_block_size': avg_block_size,
            'block_sizes': block_sizes
        }

