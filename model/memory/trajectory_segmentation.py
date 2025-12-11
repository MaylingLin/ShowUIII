"""
Trajectory Segmentation Module

Implements multiple strategies for segmenting long trajectories into
semantically coherent blocks.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseSegmenter(ABC):
    """Base class for trajectory segmenters"""
    
    @abstractmethod
    def segment(self, trajectory: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Segment a trajectory into blocks.
        
        Args:
            trajectory: List of steps, each step is a dict with keys:
                - 'obs': observation (image path or PIL Image)
                - 'action': dict with 'action', 'position', 'value'
                - 'reward': optional reward value
        
        Returns:
            List of blocks, each block is a list of steps
        """
        pass


class FixedWindowSegmenter(BaseSegmenter):
    """
    Baseline segmenter that splits trajectory into fixed-size windows.
    
    Simple implementation for ablation comparison.
    """
    
    def __init__(self, window_size: int = 4):
        """
        Args:
            window_size: Number of steps per block
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        
        self.window_size = window_size
    
    def segment(self, trajectory: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Segment trajectory into fixed-size windows"""
        if not trajectory:
            return []
        
        blocks = []
        for i in range(0, len(trajectory), self.window_size):
            block = trajectory[i:i+self.window_size]
            blocks.append(block)
        
        return blocks


class RuleBasedSegmenter(BaseSegmenter):
    """
    Rule-based segmenter that detects block boundaries based on
    action type transitions and semantic patterns.
    
    Boundary patterns are provided via constructor to avoid hardcoding.
    """
    
    def __init__(
        self,
        min_block_size: int = 2,
        max_block_size: int = 8,
        boundary_patterns: Optional[List[tuple]] = None
    ):
        """
        Args:
            min_block_size: Minimum number of steps in a block
            max_block_size: Maximum number of steps in a block (force split)
            boundary_patterns: List of (prev_action, curr_action) tuples that indicate boundaries.
                             Use '*' for wildcard. If None, uses get_default_patterns()
        """
        if min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1, got {min_block_size}")
        if max_block_size < min_block_size:
            raise ValueError(f"max_block_size ({max_block_size}) < min_block_size ({min_block_size})")
        
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.boundary_patterns = boundary_patterns if boundary_patterns is not None else self.get_default_patterns()
    
    @staticmethod
    def get_default_patterns() -> List[tuple]:
        """
        Get default boundary patterns for common GUI action spaces.
        
        These patterns are based on analysis of Mind2Web and GUIAct datasets.
        Can be overridden by passing custom patterns to constructor.
        """
        return [
            ('SCROLL', 'CLICK'),
            ('SCROLL', 'TAP'),
            ('CLICK', 'INPUT'),
            ('TAP', 'INPUT'),
            ('INPUT', 'CLICK'),
            ('INPUT', 'TAP'),
            ('INPUT', 'ENTER'),
            ('SELECT', 'CLICK'),
            ('HOVER', 'CLICK'),
            ('ANSWER', '*'),
            ('SWIPE', 'TAP'),
        ]
    
    def _normalize_action_type(self, action_type: str) -> str:
        """Normalize action type to uppercase for matching"""
        if isinstance(action_type, str):
            return action_type.upper().strip()
        return str(action_type).upper().strip()
    
    def is_boundary(self, prev_action: str, curr_action: str) -> bool:
        """
        Check if transition from prev_action to curr_action is a boundary.
        
        Args:
            prev_action: Previous action type (normalized)
            curr_action: Current action type (normalized)
        
        Returns:
            True if this transition indicates a block boundary
        """
        for pattern_prev, pattern_curr in self.boundary_patterns:
            if pattern_prev == prev_action:
                if pattern_curr == '*' or pattern_curr == curr_action:
                    return True
        return False
    
    def segment(self, trajectory: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Segment trajectory based on action transition rules"""
        if not trajectory:
            return []
        
        if len(trajectory) == 1:
            return [trajectory]
        
        blocks = []
        current_block = [trajectory[0]]
        
        for i in range(1, len(trajectory)):
            prev_step = trajectory[i-1]
            curr_step = trajectory[i]
            
            # Extract action types (handle different structures)
            if isinstance(prev_step.get('action'), dict):
                prev_action = prev_step['action'].get('action', '')
            else:
                prev_action = str(prev_step.get('action', ''))
            
            if isinstance(curr_step.get('action'), dict):
                curr_action = curr_step['action'].get('action', '')
            else:
                curr_action = str(curr_step.get('action', ''))
            
            prev_action = self._normalize_action_type(prev_action)
            curr_action = self._normalize_action_type(curr_action)
            
            should_split = False
            
            if (self.is_boundary(prev_action, curr_action) and 
                len(current_block) >= self.min_block_size):
                should_split = True
            elif len(current_block) >= self.max_block_size:
                should_split = True
            
            if should_split:
                blocks.append(current_block)
                current_block = [curr_step]
            else:
                current_block.append(curr_step)
        
        if current_block:
            blocks.append(current_block)
        
        return blocks


def create_segmenter(segmenter_type: str, **kwargs) -> BaseSegmenter:
    """
    Factory function to create segmenter by type.
    
    Args:
        segmenter_type: 'fixed' or 'rule'
        **kwargs: Arguments passed to segmenter constructor
    
    Returns:
        BaseSegmenter instance
    """
    if segmenter_type == 'fixed':
        return FixedWindowSegmenter(**kwargs)
    elif segmenter_type == 'rule':
        return RuleBasedSegmenter(**kwargs)
    else:
        raise ValueError(f"Unknown segmenter type: {segmenter_type}. "
                        f"Choose from: 'fixed', 'rule'")

