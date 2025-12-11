"""
Hierarchical Memory System for GUI Agent

This module implements trajectory block segmentation and encoding
for long-sequence GUI interaction tasks.
"""

from .trajectory_segmentation import (
    FixedWindowSegmenter,
    RuleBasedSegmenter,
    BaseSegmenter,
    create_segmenter
)
from .block_encoder import TrajectoryBlockEncoder
from .losses import BlockEncoderLoss

__all__ = [
    'FixedWindowSegmenter',
    'RuleBasedSegmenter',
    'BaseSegmenter',
    'TrajectoryBlockEncoder',
    'BlockEncoderLoss',
    'create_segmenter'
]

