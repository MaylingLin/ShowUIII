"""
Comprehensive Test Script for Three-Layer Memory System

Tests low-layer, mid-layer, high-layer memory and cross-layer retrieval.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from typing import List, Dict

from model.memory import (
    LowLayerMemory,
    MidLayerMemory,
    HighLayerMemory,
    TrajectoryBlockEncoder,
    CrossLayerRetrieval,
    HierarchicalMemoryManager
)


def test_low_layer_memory():
    """Test low-layer memory FIFO buffer"""
    print("\n" + "="*70)
    print("TEST 1: Low-Layer Memory")
    print("="*70)
    
    # Initialize
    low_mem = LowLayerMemory(max_size=5, vision_hidden_size=128)
    
    print(f"\n1.1 Initial state: {low_mem}")
    assert len(low_mem) == 0
    
    # Add steps
    print(f"\n1.2 Adding 7 steps (max_size=5, should keep last 5)")
    for i in range(7):
        vision_feat = torch.randn(1, 128)
        low_mem.add_step(
            vision_features=vision_feat,
            action_type=i % 3,
            action_position=[0.1 * i, 0.2 * i],
            reward=float(i)
        )
    
    print(f"    Current size: {len(low_mem)}")
    assert len(low_mem) == 5, f"Expected 5, got {len(low_mem)}"
    
    # Get recent steps
    print(f"\n1.3 Retrieving recent 3 steps")
    recent = low_mem.get_recent_steps(k=3)
    print(f"    Got {len(recent)} steps")
    assert len(recent) == 3
    
    # Get stats
    stats = low_mem.get_stats()
    print(f"\n1.4 Memory stats:")
    print(f"    Size: {stats['size']}/{stats['max_size']}")
    print(f"    Total steps: {stats['total_steps']}")
    print(f"    Action types: {stats['action_types']}")
    
    # Clear
    low_mem.clear()
    assert len(low_mem) == 0
    print(f"\n1.5 After clear: {len(low_mem)} steps")
    
    print("\n✓ Low-layer memory tests passed!")
    return low_mem


def test_mid_layer_memory():
    """Test mid-layer memory with semantic indexing"""
    print("\n" + "="*70)
    print("TEST 2: Mid-Layer Memory")
    print("="*70)
    
    # Initialize
    mid_mem = MidLayerMemory(block_hidden_dim=256, max_blocks=10)
    
    print(f"\n2.1 Initial state: {mid_mem}")
    assert len(mid_mem) == 0
    
    # Add blocks
    print(f"\n2.2 Adding 5 blocks")
    block_ids = []
    for i in range(5):
        block_emb = torch.randn(1, 256)
        metadata = {
            'actions': [0, 1, 2],
            'positions': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            'rewards': [1.0, 0.5, 0.0],
            'start_step': i * 3,
            'end_step': (i + 1) * 3,
            'timestamp': (i + 1) * 3
        }
        block_id = mid_mem.add_block(block_emb, metadata)
        block_ids.append(block_id)
    
    print(f"    Current size: {len(mid_mem)}")
    print(f"    Block IDs: {block_ids}")
    assert len(mid_mem) == 5
    
    # Retrieve similar blocks
    print(f"\n2.3 Retrieving top-3 similar blocks")
    query_emb = torch.randn(1, 256)
    similar = mid_mem.retrieve_similar_blocks(query_emb, top_k=3)
    print(f"    Retrieved {len(similar)} blocks:")
    for block_id, score, meta in similar:
        print(f"      Block {block_id}: score={score:.4f}, actions={meta['actions']}")
    
    # Get recent blocks
    print(f"\n2.4 Getting recent 2 blocks")
    recent = mid_mem.get_recent_blocks(k=2)
    print(f"    Got {len(recent)} blocks")
    
    # Stats
    stats = mid_mem.get_stats()
    print(f"\n2.5 Memory stats:")
    print(f"    Size: {stats['size']}/{stats['max_blocks']}")
    print(f"    Avg block size: {stats['avg_block_size']:.2f}")
    
    print("\n✓ Mid-layer memory tests passed!")
    return mid_mem


def test_high_layer_memory():
    """Test high-layer memory with task tracking"""
    print("\n" + "="*70)
    print("TEST 3: High-Layer Memory")
    print("="*70)
    
    # Initialize
    high_mem = HighLayerMemory(task_hidden_dim=512, max_subgoals=5)
    
    print(f"\n3.1 Initial state: {high_mem}")
    
    # Set task intent
    print(f"\n3.2 Setting task intent")
    task_desc = "Search for a product and add to cart"
    task_emb = torch.randn(1, 512)
    high_mem.set_task_intent(task_desc, task_emb)
    
    context = high_mem.get_task_context()
    print(f"    Task: {context['task_description']}")
    print(f"    Has intent: {context['task_intent'] is not None}")
    
    # Add subgoals
    print(f"\n3.3 Adding 3 subgoals")
    subgoals = [
        ("Search for product", None, 5),
        ("Select best match", None, 3),
        ("Add to cart", None, 2)
    ]
    
    for desc, emb, steps in subgoals:
        idx = high_mem.add_subgoal(desc, emb, steps)
        print(f"    Subgoal {idx}: {desc} ({steps} steps)")
    
    # Update subgoal status
    print(f"\n3.4 Updating subgoal statuses")
    high_mem.update_subgoal_status(0, 'in_progress', current_step=0)
    high_mem.update_subgoal_status(0, 'completed', current_step=5)
    high_mem.update_subgoal_status(1, 'in_progress', current_step=5)
    
    # Get context
    context = high_mem.get_task_context()
    print(f"    Progress: {context['progress']:.2f}")
    print(f"    Completed: {context['completed_subgoals']}/{context['num_subgoals']}")
    print(f"    Current subgoal: {context['current_subgoal']['description'] if context['current_subgoal'] else 'None'}")
    
    # Stats
    stats = high_mem.get_stats()
    print(f"\n3.5 Memory stats:")
    print(f"    Subgoals: {stats['num_subgoals']}")
    print(f"    Statuses: {stats['subgoal_statuses']}")
    
    print("\n✓ High-layer memory tests passed!")
    return high_mem


def test_cross_layer_retrieval():
    """Test cross-layer retrieval mechanism"""
    print("\n" + "="*70)
    print("TEST 4: Cross-Layer Retrieval")
    print("="*70)
    
    # Initialize retrieval module
    retrieval = CrossLayerRetrieval(
        query_dim=128,
        low_layer_dim=128,
        mid_layer_dim=256,
        high_layer_dim=512,
        output_dim=512,
        num_heads=8
    )
    
    print(f"\n4.1 Initialized CrossLayerRetrieval")
    print(f"    Parameters: {retrieval.get_num_params():,}")
    
    # Prepare test data
    print(f"\n4.2 Preparing test memories")
    
    # Low-layer memory
    low_steps = []
    for i in range(5):
        low_steps.append({
            'vision_features': torch.randn(1, 128),
            'action_type': i % 3,
            'timestamp': i
        })
    print(f"    Low-layer: {len(low_steps)} steps")
    
    # Mid-layer memory
    mid_blocks = [torch.randn(1, 256) for _ in range(3)]
    print(f"    Mid-layer: {len(mid_blocks)} blocks")
    
    # High-layer memory
    high_context = {
        'task_intent': torch.randn(1, 512),
        'task_description': 'Test task',
        'progress': 0.5
    }
    print(f"    High-layer: Task intent set")
    
    # Test retrieval
    print(f"\n4.3 Testing retrieval")
    current_obs = torch.randn(2, 128)  # Batch size 2
    
    with torch.no_grad():
        memory_context, info = retrieval(
            current_obs,
            low_steps,
            mid_blocks,
            high_context
        )
    
    print(f"    Memory context shape: {memory_context.shape}")
    print(f"    Layer weights: {info['layer_weights'].numpy()}")
    print(f"      Low: {info['layer_weights'][0, 0]:.3f}")
    print(f"      Mid: {info['layer_weights'][0, 1]:.3f}")
    print(f"      High: {info['layer_weights'][0, 2]:.3f}")
    
    assert memory_context.shape == (2, 512)
    
    print("\n✓ Cross-layer retrieval tests passed!")
    return retrieval


def test_hierarchical_memory_manager():
    """Test complete memory manager"""
    print("\n" + "="*70)
    print("TEST 5: Hierarchical Memory Manager (Integration)")
    print("="*70)
    
    # Initialize components
    print(f"\n5.1 Initializing components")
    
    block_encoder = TrajectoryBlockEncoder(
        vision_hidden_size=128,
        action_vocab_size=10,
        position_dim=4,
        hidden_dim=256,
        num_layers=2
    )
    print(f"    Block encoder: {block_encoder.get_num_params():,} params")
    
    retrieval = CrossLayerRetrieval(
        query_dim=128,
        low_layer_dim=128,
        mid_layer_dim=256,
        high_layer_dim=256,
        output_dim=512,
        num_heads=8
    )
    print(f"    Retrieval: {retrieval.get_num_params():,} params")
    
    # Initialize manager
    manager = HierarchicalMemoryManager(
        low_memory_max_size=10,
        mid_memory_max_blocks=20,
        high_memory_max_subgoals=5,
        block_encoder=block_encoder,
        cross_layer_retrieval=retrieval,
        segmenter_type='fixed',
        segmenter_config={'window_size': 3},
        min_block_size=2,
        max_block_size=4
    )
    
    print(f"\n5.2 Manager initialized: {manager}")
    
    # Start episode
    print(f"\n5.3 Starting episode")
    task_desc = "Navigate to settings and change theme"
    task_emb = torch.randn(1, 256)
    manager.on_episode_start(task_desc, task_emb)
    
    # Simulate steps
    print(f"\n5.4 Simulating 10 steps")
    for i in range(10):
        vision_feat = torch.randn(128)
        action_type = i % 5
        action_pos = [np.random.rand(), np.random.rand()]
        reward = 1.0 if i % 3 == 0 else 0.0
        
        manager.on_step(
            vision_features=vision_feat,
            action_type=action_type,
            action_position=action_pos,
            reward=reward
        )
    
    # Check memory state
    stats = manager.get_memory_stats()
    print(f"\n5.5 Memory state after 10 steps:")
    print(f"    Low-layer: {stats['low_memory']['size']} steps")
    print(f"    Mid-layer: {stats['mid_memory']['size']} blocks")
    print(f"    Pending: {stats['pending_steps']} steps")
    
    # Test retrieval
    print(f"\n5.6 Testing memory retrieval")
    current_obs = torch.randn(1, 128)
    
    with torch.no_grad():
        memory_context, info = manager.retrieve_memory_context(current_obs)
    
    print(f"    Retrieved context shape: {memory_context.shape}")
    print(f"    Layer weights: {info['layer_weights'][0].numpy()}")
    
    # End episode
    manager.on_episode_end(success=True)
    
    print(f"\n5.7 Final stats:")
    final_stats = manager.get_memory_stats()
    print(f"    Total blocks created: {final_stats['total_blocks_created']}")
    print(f"    Episodes: {final_stats['episode_count']}")
    
    print("\n✓ Hierarchical memory manager tests passed!")
    return manager


def test_empty_memory_handling():
    """Test handling of empty memories"""
    print("\n" + "="*70)
    print("TEST 6: Empty Memory Handling")
    print("="*70)
    
    retrieval = CrossLayerRetrieval(
        query_dim=128,
        low_layer_dim=128,
        mid_layer_dim=256,
        high_layer_dim=512,
        output_dim=512,
        num_heads=8
    )
    
    # Test with empty memories
    print(f"\n6.1 Testing with all empty memories")
    current_obs = torch.randn(1, 128)
    
    with torch.no_grad():
        memory_context, info = retrieval(
            current_obs,
            [],  # Empty low
            [],  # Empty mid
            {'task_intent': None}  # Empty high
        )
    
    print(f"    Context shape: {memory_context.shape}")
    print(f"    All zeros: {torch.allclose(memory_context, torch.zeros_like(memory_context), atol=1e-5)}")
    
    print("\n✓ Empty memory handling tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Three-Layer Memory System Comprehensive Test Suite")
    print("="*70)
    
    try:
        # Test individual layers
        low_mem = test_low_layer_memory()
        mid_mem = test_mid_layer_memory()
        high_mem = test_high_layer_memory()
        
        # Test retrieval
        retrieval = test_cross_layer_retrieval()
        
        # Test integration
        manager = test_hierarchical_memory_manager()
        
        # Test edge cases
        test_empty_memory_handling()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        
        print("\nThree-layer memory system is ready for integration!")
        print("\nNext steps:")
        print("  1. Integrate with data loading pipeline")
        print("  2. Add to training loop")
        print("  3. Run ablation experiments")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

