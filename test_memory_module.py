"""
Comprehensive Test Script for Memory Module

Tests trajectory segmentation and block encoding functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
from PIL import Image
import numpy as np

from model.memory import (
    FixedWindowSegmenter,
    RuleBasedSegmenter,
    TrajectoryBlockEncoder,
    BlockEncoderLoss,
    create_segmenter
)
from model.memory.action_vocab import ActionVocabulary, get_action_vocab_for_dataset


def test_segmenters():
    """Test trajectory segmentation"""
    print("\n" + "="*60)
    print("TEST 1: Trajectory Segmentation")
    print("="*60)
    
    # Create sample trajectory
    sample_trajectory = [
        {'obs': 'img1.png', 'action': {'action': 'SCROLL', 'position': [0.5, 0.3], 'value': None}},
        {'obs': 'img2.png', 'action': {'action': 'SCROLL', 'position': [0.5, 0.5], 'value': None}},
        {'obs': 'img3.png', 'action': {'action': 'CLICK', 'position': [0.6, 0.4], 'value': None}},
        {'obs': 'img4.png', 'action': {'action': 'INPUT', 'position': [0.7, 0.5], 'value': 'test'}},
        {'obs': 'img5.png', 'action': {'action': 'INPUT', 'position': [0.7, 0.5], 'value': 'query'}},
        {'obs': 'img6.png', 'action': {'action': 'CLICK', 'position': [0.8, 0.6], 'value': None}},
        {'obs': 'img7.png', 'action': {'action': 'SCROLL', 'position': [0.5, 0.7], 'value': None}},
        {'obs': 'img8.png', 'action': {'action': 'TAP', 'position': [0.4, 0.5], 'value': None}},
    ]
    
    # Test Fixed Window Segmenter
    print("\n1.1 Fixed Window Segmenter (window_size=3)")
    fixed_seg = FixedWindowSegmenter(window_size=3)
    fixed_blocks = fixed_seg.segment(sample_trajectory)
    print(f"   Number of blocks: {len(fixed_blocks)}")
    for i, block in enumerate(fixed_blocks):
        actions = [step['action']['action'] for step in block]
        print(f"   Block {i+1}: {actions}")
    
    # Test Rule-Based Segmenter
    print("\n1.2 Rule-Based Segmenter")
    rule_seg = RuleBasedSegmenter(min_block_size=2, max_block_size=5)
    rule_blocks = rule_seg.segment(sample_trajectory)
    print(f"   Number of blocks: {len(rule_blocks)}")
    for i, block in enumerate(rule_blocks):
        actions = [step['action']['action'] for step in block]
        print(f"   Block {i+1}: {actions}")
    
    # Test with custom patterns
    print("\n1.3 Rule-Based with Custom Patterns")
    custom_patterns = [('SCROLL', 'CLICK'), ('INPUT', 'CLICK')]
    custom_seg = RuleBasedSegmenter(
        min_block_size=2,
        max_block_size=6,
        boundary_patterns=custom_patterns
    )
    custom_blocks = custom_seg.segment(sample_trajectory)
    print(f"   Number of blocks: {len(custom_blocks)}")
    for i, block in enumerate(custom_blocks):
        actions = [step['action']['action'] for step in block]
        print(f"   Block {i+1}: {actions}")
    
    # Test factory function
    print("\n1.4 Factory Function")
    seg_fixed = create_segmenter('fixed', window_size=4)
    seg_rule = create_segmenter('rule', min_block_size=2)
    print(f"   Created fixed segmenter: {type(seg_fixed).__name__}")
    print(f"   Created rule segmenter: {type(seg_rule).__name__}")
    
    print("\n✓ Segmentation tests passed!")
    return rule_blocks


def test_block_encoder(blocks):
    """Test block encoder"""
    print("\n" + "="*60)
    print("TEST 2: Block Encoder")
    print("="*60)
    
    # Configuration (these should come from model config in real usage)
    vision_hidden_size = 1536  # Qwen2-VL vision encoder output
    action_vocab_size = 15  # Example vocab size
    batch_size = 2
    num_steps = 4
    
    print(f"\nConfiguration:")
    print(f"   Vision hidden size: {vision_hidden_size}")
    print(f"   Action vocab size: {action_vocab_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per block: {num_steps}")
    
    # Test both compression methods
    for compression_method in ['query', 'mean']:
        print(f"\n2.1 Testing compression method: {compression_method}")
        
        encoder = TrajectoryBlockEncoder(
            vision_hidden_size=vision_hidden_size,
            action_vocab_size=action_vocab_size,
            position_dim=4,
            hidden_dim=512,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            compression_method=compression_method,
            max_block_length=10
        )
        
        print(f"   Total parameters: {encoder.get_num_params():,}")
        
        # Create dummy input
        vision_features = torch.randn(batch_size, num_steps, vision_hidden_size)
        action_types = torch.randint(0, action_vocab_size, (batch_size, num_steps))
        action_positions = torch.rand(batch_size, num_steps, 4)  # Normalized coordinates
        attention_mask = torch.ones(batch_size, num_steps)
        
        # Forward pass
        with torch.no_grad():
            block_embs = encoder(
                vision_features,
                action_types,
                action_positions,
                attention_mask
            )
        
        print(f"   Input shape: {vision_features.shape}")
        print(f"   Output shape: {block_embs.shape}")
        print(f"   Output norm: {block_embs.norm(dim=-1).mean().item():.4f}")
        
        # Test with padding
        print(f"\n2.2 Testing with padding")
        attention_mask[1, 2:] = 0  # Mask out last 2 steps of second sample
        
        with torch.no_grad():
            block_embs_padded = encoder(
                vision_features,
                action_types,
                action_positions,
                attention_mask
            )
        
        print(f"   Output shape with padding: {block_embs_padded.shape}")
        print(f"   Output norm: {block_embs_padded.norm(dim=-1).mean().item():.4f}")
    
    print("\n✓ Block encoder tests passed!")
    return encoder, (vision_features, action_types, action_positions, attention_mask)


def test_losses(encoder, data):
    """Test loss functions"""
    print("\n" + "="*60)
    print("TEST 3: Loss Functions")
    print("="*60)
    
    vision_features, action_types, action_positions, attention_mask = data
    action_vocab_size = encoder.action_vocab_size
    batch_size = vision_features.shape[0]
    
    # Compute block embeddings
    with torch.no_grad():
        block_embs = encoder(vision_features, action_types, action_positions, attention_mask)
    
    # Test action presence loss
    print("\n3.1 Action Presence Loss")
    loss_module = BlockEncoderLoss(
        action_vocab_size=action_vocab_size,
        hidden_dim=encoder.hidden_dim,
        use_contrastive=False
    )
    
    loss_dict = loss_module(
        block_embs,
        action_types,
        attention_mask
    )
    
    print(f"   Action loss: {loss_dict['action_loss'].item():.4f}")
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    
    # Test with contrastive loss
    print("\n3.2 With Contrastive Loss")
    loss_module_cont = BlockEncoderLoss(
        action_vocab_size=action_vocab_size,
        hidden_dim=encoder.hidden_dim,
        use_contrastive=True,
        contrastive_temperature=0.07,
        action_loss_weight=1.0,
        contrastive_loss_weight=0.1
    )
    
    task_ids = torch.tensor([0, 0])  # Same task
    loss_dict_cont = loss_module_cont(
        block_embs,
        action_types,
        attention_mask,
        task_ids
    )
    
    print(f"   Action loss: {loss_dict_cont['action_loss'].item():.4f}")
    print(f"   Contrastive loss: {loss_dict_cont['contrastive_loss'].item():.4f}")
    print(f"   Total loss: {loss_dict_cont['total_loss'].item():.4f}")
    
    # Test backward pass
    print("\n3.3 Backward Pass Test")
    loss = loss_dict_cont['total_loss']
    loss.backward()
    print(f"   ✓ Backward pass successful")
    
    # Check gradients
    has_grad = sum(1 for p in loss_module_cont.parameters() if p.grad is not None)
    total_params = sum(1 for p in loss_module_cont.parameters())
    print(f"   Parameters with gradients: {has_grad}/{total_params}")
    
    print("\n✓ Loss function tests passed!")


def test_action_vocabulary():
    """Test action vocabulary builder"""
    print("\n" + "="*60)
    print("TEST 4: Action Vocabulary")
    print("="*60)
    
    # Create sample vocabulary
    vocab = ActionVocabulary()
    
    # Manually add some actions for testing
    vocab.action_to_idx = {
        '<PAD>': 0,
        'CLICK': 1,
        'INPUT': 2,
        'SCROLL': 3,
        'TAP': 4,
        'ANSWER': 5
    }
    vocab.idx_to_action = {v: k for k, v in vocab.action_to_idx.items()}
    
    print(f"\n4.1 Vocabulary Operations")
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Actions: {list(vocab.action_to_idx.keys())}")
    
    # Test encoding
    print(f"\n4.2 Encoding")
    test_actions = ['CLICK', 'click', 'INPUT', 'UNKNOWN']
    for action in test_actions:
        idx = vocab.encode(action)
        decoded = vocab.decode(idx)
        print(f"   '{action}' -> {idx} -> '{decoded}'")
    
    # Test save/load
    print(f"\n4.3 Save/Load")
    test_path = '/tmp/test_action_vocab.json'
    vocab.save(test_path)
    print(f"   Saved to {test_path}")
    
    vocab_loaded = ActionVocabulary()
    vocab_loaded.load(test_path)
    print(f"   Loaded vocabulary size: {len(vocab_loaded)}")
    print(f"   Vocabularies match: {vocab.action_to_idx == vocab_loaded.action_to_idx}")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("\n✓ Action vocabulary tests passed!")


def test_integration():
    """Integration test: full pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Integration Test")
    print("="*60)
    
    # Simulated trajectory
    trajectory = [
        {'obs': 'img1.png', 'action': {'action': 'SCROLL', 'position': [0.5, 0.3], 'value': None}},
        {'obs': 'img2.png', 'action': {'action': 'CLICK', 'position': [0.6, 0.4], 'value': None}},
        {'obs': 'img3.png', 'action': {'action': 'INPUT', 'position': [0.7, 0.5], 'value': 'query'}},
        {'obs': 'img4.png', 'action': {'action': 'CLICK', 'position': [0.8, 0.6], 'value': None}},
    ]
    
    print("\n5.1 Segment Trajectory")
    segmenter = RuleBasedSegmenter(min_block_size=2, max_block_size=4)
    blocks = segmenter.segment(trajectory)
    print(f"   Segmented into {len(blocks)} blocks")
    
    print("\n5.2 Encode Blocks")
    # Create encoder
    encoder = TrajectoryBlockEncoder(
        vision_hidden_size=1536,
        action_vocab_size=10,
        hidden_dim=256,
        num_layers=2
    )
    
    # Create action vocabulary
    action_vocab = ActionVocabulary()
    action_vocab.action_to_idx = {
        '<PAD>': 0, 'CLICK': 1, 'INPUT': 2, 'SCROLL': 3,
        'TAP': 4, 'ANSWER': 5, 'ENTER': 6
    }
    action_vocab.idx_to_action = {v: k for k, v in action_vocab.action_to_idx.items()}
    
    # Encode first block
    block = blocks[0]
    num_steps = len(block)
    
    # Create dummy vision features
    vision_features = torch.randn(1, num_steps, 1536)
    action_types = torch.tensor([[action_vocab.encode(step['action']['action']) for step in block]])
    action_positions = torch.tensor([[[*step['action']['position'], 0.0, 0.0] for step in block]])
    
    with torch.no_grad():
        block_emb = encoder(vision_features, action_types, action_positions)
    
    print(f"   Block embedding shape: {block_emb.shape}")
    print(f"   Block embedding norm: {block_emb.norm().item():.4f}")
    
    print("\n✓ Integration test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Memory Module Comprehensive Test Suite")
    print("="*60)
    
    try:
        # Test 1: Segmentation
        blocks = test_segmenters()
        
        # Test 2: Block Encoder
        encoder, data = test_block_encoder(blocks)
        
        # Test 3: Loss Functions
        test_losses(encoder, data)
        
        # Test 4: Action Vocabulary
        test_action_vocabulary()
        
        # Test 5: Integration
        test_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nMemory module is ready for integration into training pipeline.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

