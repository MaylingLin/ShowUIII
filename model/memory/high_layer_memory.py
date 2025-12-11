"""
High-Layer Memory Module

Stores task-level abstractions including task intent and subgoal tracking.
"""

from typing import List, Dict, Optional
import torch


class HighLayerMemory:
    """
    High-layer memory: Stores task-level abstractions
    
    Features:
    - Global perspective: Task intent and subgoals
    - Long-term dependencies: Cross-episode patterns (optional)
    - Sparse updates: Only at key milestones
    """
    
    def __init__(
        self,
        task_hidden_dim: int,
        max_subgoals: int = 10,
        store_on_cpu: bool = True
    ):
        """
        Args:
            task_hidden_dim: Dimension of task/subgoal embeddings
            max_subgoals: Maximum number of subgoals to track
            store_on_cpu: If True, store embeddings on CPU
        """
        if task_hidden_dim < 1:
            raise ValueError(f"task_hidden_dim must be >= 1, got {task_hidden_dim}")
        if max_subgoals < 1:
            raise ValueError(f"max_subgoals must be >= 1, got {max_subgoals}")
        
        self.task_hidden_dim = task_hidden_dim
        self.max_subgoals = max_subgoals
        self.store_on_cpu = store_on_cpu
        
        # Current episode task information
        self.task_intent = None  # (1, task_hidden_dim)
        self.task_description = None  # str
        self.subgoals = []  # List of subgoal dicts
        self.current_subgoal_idx = 0
        
        # Optional: Cross-episode long-term memory (not implemented yet)
        self.success_patterns = []
    
    def set_task_intent(
        self,
        task_description: str,
        task_embedding: torch.Tensor
    ):
        """
        Set the current task's intent vector
        
        Args:
            task_description: Natural language task description
            task_embedding: (1, task_hidden_dim) encoded task vector
        """
        # Validate
        if task_embedding.dim() != 2 or task_embedding.shape[1] != self.task_hidden_dim:
            raise ValueError(
                f"task_embedding must have shape (1, {self.task_hidden_dim}), "
                f"got {task_embedding.shape}"
            )
        
        if self.store_on_cpu and task_embedding.is_cuda:
            task_embedding = task_embedding.cpu()
        
        self.task_intent = task_embedding.detach()
        self.task_description = task_description
    
    def add_subgoal(
        self,
        subgoal_description: str,
        subgoal_embedding: Optional[torch.Tensor] = None,
        estimated_steps: int = 5
    ) -> int:
        """
        Add a subgoal to the tracking list
        
        Args:
            subgoal_description: Natural language subgoal description
            subgoal_embedding: Optional (1, task_hidden_dim) subgoal vector
            estimated_steps: Estimated number of steps to complete
        
        Returns:
            subgoal_idx: Index of the added subgoal
        """
        if len(self.subgoals) >= self.max_subgoals:
            raise ValueError(f"Maximum subgoals ({self.max_subgoals}) reached")
        
        # Validate embedding if provided
        if subgoal_embedding is not None:
            if subgoal_embedding.dim() != 2 or subgoal_embedding.shape[1] != self.task_hidden_dim:
                raise ValueError(
                    f"subgoal_embedding must have shape (1, {self.task_hidden_dim}), "
                    f"got {subgoal_embedding.shape}"
                )
            
            if self.store_on_cpu and subgoal_embedding.is_cuda:
                subgoal_embedding = subgoal_embedding.cpu()
            
            subgoal_embedding = subgoal_embedding.detach()
        
        subgoal = {
            'description': subgoal_description,
            'embedding': subgoal_embedding,
            'estimated_steps': estimated_steps,
            'actual_steps': 0,
            'status': 'pending',  # pending/in_progress/completed/failed
            'start_step': None,
            'end_step': None
        }
        
        self.subgoals.append(subgoal)
        return len(self.subgoals) - 1
    
    def update_subgoal_status(
        self,
        subgoal_idx: int,
        status: str,
        current_step: int
    ):
        """
        Update subgoal status
        
        Args:
            subgoal_idx: Index of subgoal to update
            status: New status ('pending', 'in_progress', 'completed', 'failed')
            current_step: Current step number
        """
        valid_statuses = ['pending', 'in_progress', 'completed', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}, got {status}")
        
        if 0 <= subgoal_idx < len(self.subgoals):
            subgoal = self.subgoals[subgoal_idx]
            subgoal['status'] = status
            
            # Update timestamps
            if status == 'in_progress' and subgoal['start_step'] is None:
                subgoal['start_step'] = current_step
            elif status in ['completed', 'failed']:
                subgoal['end_step'] = current_step
                if subgoal['start_step'] is not None:
                    subgoal['actual_steps'] = current_step - subgoal['start_step']
            
            # Move to next subgoal if completed
            if status == 'completed' and subgoal_idx == self.current_subgoal_idx:
                self.current_subgoal_idx = min(subgoal_idx + 1, len(self.subgoals))
    
    def get_current_subgoal(self) -> Optional[Dict]:
        """
        Get the currently active subgoal
        
        Returns:
            Subgoal dict or None if all completed
        """
        if 0 <= self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return None
    
    def get_task_context(self) -> Dict:
        """
        Get complete task-level context
        
        Returns:
            Dict with keys:
                - 'task_intent': tensor or None
                - 'task_description': str or None
                - 'current_subgoal': dict or None
                - 'progress': float (0-1)
                - 'num_subgoals': int
                - 'completed_subgoals': int
                - 'failed_subgoals': int
        """
        completed = sum(1 for sg in self.subgoals if sg['status'] == 'completed')
        failed = sum(1 for sg in self.subgoals if sg['status'] == 'failed')
        progress = completed / len(self.subgoals) if self.subgoals else 0.0
        
        return {
            'task_intent': self.task_intent,
            'task_description': self.task_description,
            'current_subgoal': self.get_current_subgoal(),
            'progress': progress,
            'num_subgoals': len(self.subgoals),
            'completed_subgoals': completed,
            'failed_subgoals': failed,
            'all_subgoals': self.subgoals
        }
    
    def clear(self):
        """Clear high-layer memory (new episode)"""
        self.task_intent = None
        self.task_description = None
        self.subgoals = []
        self.current_subgoal_idx = 0
    
    def __repr__(self):
        completed = sum(1 for sg in self.subgoals if sg['status'] == 'completed')
        return f"HighLayerMemory(subgoals={completed}/{len(self.subgoals)}, task={self.task_description[:30] if self.task_description else 'None'}...)"
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.subgoals:
            return {
                'has_task': self.task_intent is not None,
                'task_description': self.task_description,
                'num_subgoals': 0,
                'subgoal_statuses': {}
            }
        
        # Count statuses
        status_counts = {}
        for subgoal in self.subgoals:
            status = subgoal['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'has_task': self.task_intent is not None,
            'task_description': self.task_description,
            'num_subgoals': len(self.subgoals),
            'current_subgoal_idx': self.current_subgoal_idx,
            'subgoal_statuses': status_counts,
            'subgoal_descriptions': [sg['description'] for sg in self.subgoals]
        }

