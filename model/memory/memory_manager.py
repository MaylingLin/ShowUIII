"""
Hierarchical Memory Manager

Unified interface for managing three-layer memory system.
"""

import torch
from typing import List, Dict, Tuple, Optional
import logging

from .low_layer_memory import LowLayerMemory
from .mid_layer_memory import MidLayerMemory
from .high_layer_memory import HighLayerMemory
from .block_encoder import TrajectoryBlockEncoder
from .cross_layer_retrieval import CrossLayerRetrieval
from .trajectory_segmentation import create_segmenter, BaseSegmenter


logger = logging.getLogger(__name__)


class HierarchicalMemoryManager:
    """
    Hierarchical memory manager: Unified interface for three-layer memory
    
    Features:
    - Automatic updates: Low -> Mid -> High incremental updates
    - Unified retrieval: Single call retrieves from all layers
    - Lifecycle management: Episode start/end handling
    """
    
    def __init__(
        self,
        # Memory configurations
        low_memory_max_size: int = 20,
        mid_memory_max_blocks: int = 100,
        high_memory_max_subgoals: int = 10,
        
        # Block encoder and retrieval
        block_encoder: TrajectoryBlockEncoder = None,
        cross_layer_retrieval: CrossLayerRetrieval = None,
        
        # Segmentation strategy
        segmenter_type: str = 'rule',
        segmenter_config: Optional[Dict] = None,
        
        # Auto-blocking parameters
        min_block_size: int = 2,
        max_block_size: int = 8,
        
        # Device settings
        store_on_cpu: bool = True
    ):
        """
        Args:
            low_memory_max_size: Max steps in low-layer buffer
            mid_memory_max_blocks: Max blocks in mid-layer pool
            high_memory_max_subgoals: Max subgoals to track
            block_encoder: TrajectoryBlockEncoder instance
            cross_layer_retrieval: CrossLayerRetrieval instance
            segmenter_type: 'fixed' or 'rule'
            segmenter_config: Config dict for segmenter
            min_block_size: Minimum steps per block
            max_block_size: Maximum steps per block (force split)
            store_on_cpu: Store memory on CPU to save GPU memory
        """
        # Validate required components
        if block_encoder is None:
            raise ValueError("block_encoder is required")
        if cross_layer_retrieval is None:
            raise ValueError("cross_layer_retrieval is required")
        
        # Initialize three-layer memory
        self.low_memory = LowLayerMemory(
            max_size=low_memory_max_size,
            vision_hidden_size=block_encoder.vision_hidden_size,
            store_on_cpu=store_on_cpu
        )
        
        self.mid_memory = MidLayerMemory(
            block_hidden_dim=block_encoder.hidden_dim,
            max_blocks=mid_memory_max_blocks,
            store_on_cpu=store_on_cpu
        )
        
        self.high_memory = HighLayerMemory(
            task_hidden_dim=block_encoder.hidden_dim,  # Use same dim as blocks
            max_subgoals=high_memory_max_subgoals,
            store_on_cpu=store_on_cpu
        )
        
        # Block encoder and retrieval
        self.block_encoder = block_encoder
        self.cross_layer_retrieval = cross_layer_retrieval
        
        # Segmenter
        segmenter_config = segmenter_config or {}
        self.segmenter = create_segmenter(segmenter_type, **segmenter_config)
        
        # Block creation parameters
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        
        # Pending steps buffer (for forming blocks)
        self.pending_steps = []
        self.current_step_count = 0
        
        # Statistics
        self.total_blocks_created = 0
        self.episode_count = 0
    
    def on_episode_start(
        self,
        task_description: str,
        task_embedding: torch.Tensor
    ):
        """
        Initialize memory at episode start
        
        Args:
            task_description: Natural language task description
            task_embedding: (1, task_hidden_dim) encoded task vector
        """
        # Clear all layers
        self.low_memory.clear()
        self.mid_memory.clear()
        self.high_memory.clear()
        
        # Reset pending buffer
        self.pending_steps = []
        self.current_step_count = 0
        
        # Set task intent in high-layer memory
        self.high_memory.set_task_intent(task_description, task_embedding)
        
        self.episode_count += 1
        logger.info(f"Episode {self.episode_count} started: {task_description}")
    
    def on_step(
        self,
        vision_features: torch.Tensor,
        action_type: int,
        action_position: List[float],
        action_value: Optional[str] = None,
        reward: float = 0.0
    ):
        """
        Update memory after each step
        
        Args:
            vision_features: (1, vision_hidden_size) or (vision_hidden_size,)
            action_type: Action type index
            action_position: List of coordinates
            action_value: Optional text value
            reward: Step reward
        """
        # 1. Add to low-layer memory
        self.low_memory.add_step(
            vision_features=vision_features,
            action_type=action_type,
            action_position=action_position,
            action_value=action_value,
            reward=reward,
            timestamp=self.current_step_count
        )
        
        # 2. Add to pending buffer
        self.pending_steps.append({
            'vision_features': vision_features,
            'action_type': action_type,
            'action_position': action_position,
            'action_value': action_value,
            'reward': reward,
            'timestamp': self.current_step_count
        })
        
        # 3. Check if should create new block
        self._maybe_create_block()
        
        self.current_step_count += 1
    
    def _maybe_create_block(self):
        """Decide whether to create a new block based on pending steps"""
        num_pending = len(self.pending_steps)
        
        # Force create block if reached max size
        if num_pending >= self.max_block_size:
            self._encode_and_store_block()
            return
        
        # Check if we have enough steps and should segment
        if num_pending >= self.min_block_size:
            # TODO: More sophisticated boundary detection
            # For now, use simple max_block_size threshold
            pass
    
    def _encode_and_store_block(self):
        """Encode current pending steps as a block and store to mid-layer"""
        if len(self.pending_steps) < self.min_block_size:
            return
        
        num_steps = len(self.pending_steps)
        
        try:
            # Prepare block data
            vision_features_list = []
            action_types_list = []
            action_positions_list = []
            
            for step in self.pending_steps:
                vf = step['vision_features']
                # Ensure vf is 1D (vision_hidden_size,)
                if vf.dim() == 2:
                    vf = vf.squeeze(0)  # (1, vision_hidden_size) -> (vision_hidden_size,)
                elif vf.dim() > 2:
                    vf = vf.reshape(-1)  # Flatten to 1D
                vision_features_list.append(vf)
                action_types_list.append(step['action_type'])
                
                # Ensure position has 4 dimensions (pad if necessary)
                pos = step['action_position']
                # Convert to list if it's not already
                if not isinstance(pos, list):
                    pos = list(pos) if hasattr(pos, '__iter__') else [pos]
                
                # Pad to 4 dimensions
                if len(pos) == 2:
                    pos = pos + [0.0, 0.0]  # [x, y] -> [x, y, 0, 0]
                elif len(pos) == 3:
                    pos = pos + [0.0]  # [x, y, z] -> [x, y, z, 0]
                elif len(pos) < 2:
                    pos = pos + [0.0] * (4 - len(pos))  # Pad to 4
                
                action_positions_list.append(pos[:4])  # Take first 4
            
            # Stack tensors
            vision_features = torch.stack(vision_features_list)  # (num_steps, vision_hidden_size)
            action_types = torch.tensor(action_types_list, dtype=torch.long)  # (num_steps,)
            action_positions = torch.tensor(action_positions_list, dtype=torch.float32)  # (num_steps, 4)
            
            # Add batch dimension
            vision_features = vision_features.unsqueeze(0)  # (1, num_steps, vision_hidden_size)
            action_types = action_types.unsqueeze(0)  # (1, num_steps)
            action_positions = action_positions.unsqueeze(0)  # (1, num_steps, 4)
            
            # Move to encoder's device
            device = next(self.block_encoder.parameters()).device
            vision_features = vision_features.to(device)
            action_types = action_types.to(device)
            action_positions = action_positions.to(device)
            
            # Encode block
            with torch.no_grad():
                block_embedding = self.block_encoder(
                    vision_features,
                    action_types,
                    action_positions
                )  # (1, block_hidden_dim)
            
            # Prepare metadata
            metadata = {
                'actions': [step['action_type'] for step in self.pending_steps],
                'positions': [step['action_position'] for step in self.pending_steps],
                'rewards': [step['reward'] for step in self.pending_steps],
                'start_step': self.current_step_count - num_steps,
                'end_step': self.current_step_count,
                'timestamp': self.current_step_count
            }
            
            # Store to mid-layer
            block_id = self.mid_memory.add_block(block_embedding, metadata)
            self.total_blocks_created += 1
            
            logger.debug(f"Created block {block_id} with {num_steps} steps")
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to encode block: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Clear pending buffer
        self.pending_steps = []
    
    def retrieve_memory_context(
        self,
        current_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Retrieve and fuse memory from all three layers
        
        Args:
            current_obs: (B, query_dim) current observation features
        
        Returns:
            memory_context: (B, output_dim) fused memory context
            retrieval_info: Dict with retrieval details
        """
        # Get data from three layers
        low_steps = self.low_memory.get_recent_steps(k=10)
        
        mid_blocks_data = self.mid_memory.get_recent_blocks(k=5)
        mid_blocks = [emb for emb, _ in mid_blocks_data] if mid_blocks_data else []
        
        high_context = self.high_memory.get_task_context()
        
        # Cross-layer retrieval
        memory_context, retrieval_info = self.cross_layer_retrieval(
            current_obs,
            low_steps,
            mid_blocks,
            high_context
        )
        
        return memory_context, retrieval_info
    
    def on_episode_end(self, success: bool = False):
        """
        Handle episode end
        
        Args:
            success: Whether the episode was successful
        """
        # Encode remaining pending steps
        if self.pending_steps:
            self._encode_and_store_block()
        
        logger.info(
            f"Episode {self.episode_count} ended. "
            f"Success: {success}, "
            f"Steps: {self.current_step_count}, "
            f"Blocks created: {self.total_blocks_created}"
        )
        
        # Optional: Extract success patterns (TODO for future)
        if success:
            pass  # Could save successful episodes' patterns
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        return {
            'episode_count': self.episode_count,
            'current_step': self.current_step_count,
            'total_blocks_created': self.total_blocks_created,
            'pending_steps': len(self.pending_steps),
            'low_memory': self.low_memory.get_stats(),
            'mid_memory': self.mid_memory.get_stats(),
            'high_memory': self.high_memory.get_stats()
        }
    
    def __repr__(self):
        return (
            f"HierarchicalMemoryManager("
            f"episode={self.episode_count}, "
            f"step={self.current_step_count}, "
            f"low={len(self.low_memory)}, "
            f"mid={len(self.mid_memory)}, "
            f"high={len(self.high_memory.subgoals)})"
        )

