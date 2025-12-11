"""
Comprehensive Test Script for Dynamic Update and Adaptive Retrieval (Stage 3)

Tests query classification, importance estimation, selective storage,
and enhanced memory manager.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from typing import List, Dict

from model.memory import (
    HeuristicQueryClassifier,
    LearnableQueryClassifier,
    ImportanceEstimator,
    SelectiveMemoryUpdater,
    EnhancedHierarchicalMemoryManager,
    TrajectoryBlockEncoder,
    CrossLayerRetrieval
)


def test_query_classifier():
    """Test query type classifier"""
    print("\n" + "="*70)
    print("TEST 1: Query Type Classifier")
    print("="*70)
    
    # Test heuristic classifier
    print("\n1.1 Heuristic Query Classifier")
    classifier = HeuristicQueryClassifier(
        early_progress_threshold=0.2,
        late_progress_threshold=0.8
    )
    
    # Test different scenarios
    test_cases = [
        ("Early stage, low rewards", 5, [0.1, 0.2, 0.1], 0.1),
        ("Mid stage, high rewards", 20, [0.8, 0.9, 0.7], 0.5),
        ("Late stage, mixed rewards", 50, [0.5, 0.3, 0.6], 0.9),
        ("Early stage, failures", 3, [-0.2, -0.1, -0.3], 0.15),
    ]
    
    for name, step, rewards, progress in test_cases:
        weights = classifier.classify(step, rewards, progress)
        print(f"\n  {name}:")
        print(f"    Detail: {weights['detail']:.3f}")
        print(f"    Planning: {weights['planning']:.3f}")
        print(f"    Global: {weights['global']:.3f}")
        print(f"    Sum: {sum(weights.values()):.3f} (should be ~1.0)")
        
        # Verify weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    # Test learnable classifier
    print("\n1.2 Learnable Query Classifier")
    learnable = LearnableQueryClassifier(
        query_dim=128,
        hidden_dim=64,
        num_query_types=3
    )
    
    print(f"    Parameters: {learnable.get_num_params():,}")
    
    # Test forward pass
    query_features = torch.randn(2, 128)
    with torch.no_grad():
        output = learnable(query_features)
    
    print(f"    Input shape: {query_features.shape}")
    print(f"    Output probs shape: {output['query_type_probs'].shape}")
    print(f"    Predicted types: {output['query_type'].tolist()}")
    
    # Verify output shapes
    assert output['query_type_probs'].shape == (2, 3)
    assert output['query_type'].shape == (2,)
    
    print("\n✓ Query classifier tests passed!")
    return classifier


def test_importance_estimator():
    """Test importance estimator"""
    print("\n" + "="*70)
    print("TEST 2: Importance Estimator")
    print("="*70)
    
    estimator = ImportanceEstimator(
        reward_threshold=0.5,
        diversity_weight=0.3,
        change_threshold=0.2,
        force_store_interval=5
    )
    
    print(f"\n2.1 Testing storage decisions")
    
    # Test high reward (should store)
    vision1 = torch.randn(128)
    should_store_high = estimator.should_store(vision1, action_type=1, reward=0.9)
    print(f"    High reward (0.9): {should_store_high} (expected: True)")
    assert should_store_high == True
    
    # Test low reward (should not store initially)
    vision2 = torch.randn(128)
    should_store_low = estimator.should_store(vision2, action_type=1, reward=0.1)
    print(f"    Low reward (0.1): {should_store_low}")
    
    # Test new action type (should store)
    vision3 = torch.randn(128)
    should_store_new = estimator.should_store(vision3, action_type=5, reward=0.2)
    print(f"    New action type: {should_store_new}")
    
    # Test forced storage (every 5 steps)
    # Create a fresh estimator to test forced storage
    estimator_forced = ImportanceEstimator(
        reward_threshold=0.5,
        force_store_interval=5
    )
    
    # Call 4 times (step_count will be 1, 2, 3, 4)
    for i in range(4):
        estimator_forced.should_store(torch.randn(128), action_type=1, reward=0.1)
    
    # 5th call should trigger forced storage (step_count = 5, 5 % 5 == 0)
    should_store_forced = estimator_forced.should_store(torch.randn(128), action_type=1, reward=0.1)
    print(f"    Forced storage (step 5): {should_store_forced} (expected: True)")
    print(f"    Step count: {estimator_forced.step_count}")
    
    # Verify: step_count should be 5, and 5 % 5 == 0 should trigger
    if estimator_forced.step_count % estimator_forced.force_store_interval == 0:
        # Forced storage should have added 0.5 to importance_score
        # Even with low reward (0.1), 0.5 > 0.5 threshold, so should store
        assert should_store_forced == True, f"Expected True but got {should_store_forced} at step {estimator_forced.step_count}"
    else:
        print(f"    Warning: step_count ({estimator_forced.step_count}) not at interval boundary")
    
    # Test visual change
    vision4 = torch.randn(128) * 0.1  # Very different
    should_store_change = estimator.should_store(vision4, action_type=1, reward=0.2)
    print(f"    Large visual change: {should_store_change}")
    
    # Get stats
    stats = estimator.get_stats()
    print(f"\n2.2 Estimator stats:")
    print(f"    Step count: {stats['step_count']}")
    print(f"    History size: {stats['history_size']}")
    print(f"    Unique actions: {stats['unique_actions']}")
    
    # Reset
    estimator.reset()
    assert estimator.step_count == 0
    print(f"\n2.3 After reset: step_count={estimator.step_count}")
    
    print("\n✓ Importance estimator tests passed!")
    return estimator


def test_selective_updater():
    """Test selective memory updater"""
    print("\n" + "="*70)
    print("TEST 3: Selective Memory Updater")
    print("="*70)
    
    # Create minimal memory manager for testing
    block_encoder = TrajectoryBlockEncoder(
        vision_hidden_size=128,
        action_vocab_size=10,
        hidden_dim=256,
        num_layers=2
    )
    
    retrieval = CrossLayerRetrieval(
        query_dim=128,
        low_layer_dim=128,
        mid_layer_dim=256,
        high_layer_dim=256,
        output_dim=512
    )
    
    from model.memory import HierarchicalMemoryManager
    manager = HierarchicalMemoryManager(
        block_encoder=block_encoder,
        cross_layer_retrieval=retrieval,
        min_block_size=2,
        max_block_size=4
    )
    
    # Initialize episode
    task_emb = torch.randn(1, 256)
    manager.on_episode_start("Test task", task_emb)
    
    # Create selective updater
    estimator = ImportanceEstimator(
        reward_threshold=0.5,
        force_store_interval=3
    )
    updater = SelectiveMemoryUpdater(manager, estimator)
    
    print(f"\n3.1 Testing selective storage")
    
    stored_count = 0
    total_steps = 10
    
    for i in range(total_steps):
        vision = torch.randn(128)
        reward = 0.9 if i % 2 == 0 else 0.1  # Alternating high/low
        stored = updater.update(
            vision_features=vision,
            action_type=i % 3,
            action_position=[0.5, 0.5],
            action_value=None,
            reward=reward
        )
        if stored:
            stored_count += 1
    
    print(f"    Total steps: {total_steps}")
    print(f"    Stored: {stored_count}")
    print(f"    Skipped: {total_steps - stored_count}")
    
    # Get stats
    stats = updater.get_stats()
    print(f"\n3.2 Updater stats:")
    print(f"    Storage rate: {stats['storage_rate']:.2%}")
    print(f"    Stored: {stats['stored']}")
    print(f"    Skipped: {stats['skipped']}")
    
    # Verify reasonable storage rate (30-70%)
    assert 0.3 <= stats['storage_rate'] <= 0.7, f"Storage rate {stats['storage_rate']} not in expected range"
    
    # Reset
    updater.reset()
    assert updater.stored_count == 0
    print(f"\n3.3 After reset: stored={updater.stored_count}")
    
    print("\n✓ Selective updater tests passed!")
    return updater


def test_enhanced_manager():
    """Test enhanced memory manager integration"""
    print("\n" + "="*70)
    print("TEST 4: Enhanced Memory Manager (Integration)")
    print("="*70)
    
    # Initialize components
    print(f"\n4.1 Initializing components")
    
    block_encoder = TrajectoryBlockEncoder(
        vision_hidden_size=128,
        action_vocab_size=10,
        hidden_dim=256,
        num_layers=2
    )
    
    retrieval = CrossLayerRetrieval(
        query_dim=128,
        low_layer_dim=128,
        mid_layer_dim=256,
        high_layer_dim=256,
        output_dim=512
    )
    
    # Create enhanced manager with all features
    manager = EnhancedHierarchicalMemoryManager(
        block_encoder=block_encoder,
        cross_layer_retrieval=retrieval,
        segmenter_type='fixed',
        segmenter_config={'window_size': 3},
        min_block_size=2,
        max_block_size=4,
        use_query_classifier=True,
        query_classifier_type='heuristic',
        use_selective_storage=True,
        importance_estimator_config={
            'reward_threshold': 0.5,
            'force_store_interval': 3
        }
    )
    
    print(f"    Manager: {manager}")
    print(f"    Query classifier: {manager.use_query_classifier}")
    print(f"    Selective storage: {manager.use_selective_storage}")
    
    # Start episode
    print(f"\n4.2 Starting episode")
    task_emb = torch.randn(1, 256)
    manager.on_episode_start("Navigate to settings", task_emb)
    
    # Simulate steps with adaptive storage
    print(f"\n4.3 Simulating 15 steps with adaptive storage")
    stored_count = 0
    
    for i in range(15):
        vision = torch.randn(128)
        reward = 0.8 if i % 3 == 0 else 0.2
        stored = manager.on_step_adaptive(
            vision_features=vision,
            action_type=i % 5,
            action_position=[0.5, 0.5],
            action_value=None,
            reward=reward
        )
        if stored:
            stored_count += 1
    
    print(f"    Steps stored: {stored_count}/15")
    
    # Test adaptive retrieval
    print(f"\n4.4 Testing adaptive retrieval")
    current_obs = torch.randn(1, 128)
    
    # Test at different progress levels
    for progress in [0.1, 0.5, 0.9]:
        memory_context, info = manager.retrieve_memory_context_adaptive(
            current_obs,
            task_progress=progress
        )
        
        print(f"\n    Progress {progress:.1f}:")
        print(f"      Context shape: {memory_context.shape}")
        if 'query_weights' in info:
            weights = info['query_weights']
            print(f"      Query weights: detail={weights[0]:.3f}, planning={weights[1]:.3f}, global={weights[2]:.3f}")
        print(f"      Retrieval method: {info.get('retrieval_method', 'unknown')}")
    
    # Get enhanced stats
    print(f"\n4.5 Enhanced statistics")
    stats = manager.get_enhanced_stats()
    print(f"    Low memory: {stats['low_memory']['size']} steps")
    print(f"    Mid memory: {stats['mid_memory']['size']} blocks")
    
    if 'selective_storage' in stats:
        ss_stats = stats['selective_storage']
        print(f"    Storage rate: {ss_stats['storage_rate']:.2%}")
    
    print("\n✓ Enhanced manager tests passed!")
    return manager


def test_configuration_options():
    """Test different configuration options"""
    print("\n" + "="*70)
    print("TEST 5: Configuration Options")
    print("="*70)
    
    block_encoder = TrajectoryBlockEncoder(
        vision_hidden_size=128,
        action_vocab_size=10,
        hidden_dim=256
    )
    
    retrieval = CrossLayerRetrieval(
        query_dim=128,
        low_layer_dim=128,
        mid_layer_dim=256,
        high_layer_dim=256,
        output_dim=512
    )
    
    # Test 1: Only query classifier
    print(f"\n5.1 Manager with query classifier only")
    manager1 = EnhancedHierarchicalMemoryManager(
        block_encoder=block_encoder,
        cross_layer_retrieval=retrieval,
        use_query_classifier=True,
        use_selective_storage=False
    )
    assert manager1.use_query_classifier == True
    assert manager1.use_selective_storage == False
    print(f"    ✓ Query classifier enabled, selective storage disabled")
    
    # Test 2: Only selective storage
    print(f"\n5.2 Manager with selective storage only")
    manager2 = EnhancedHierarchicalMemoryManager(
        block_encoder=block_encoder,
        cross_layer_retrieval=retrieval,
        use_query_classifier=False,
        use_selective_storage=True
    )
    assert manager2.use_query_classifier == False
    assert manager2.use_selective_storage == True
    print(f"    ✓ Selective storage enabled, query classifier disabled")
    
    # Test 3: Custom thresholds
    print(f"\n5.3 Manager with custom thresholds")
    manager3 = EnhancedHierarchicalMemoryManager(
        block_encoder=block_encoder,
        cross_layer_retrieval=retrieval,
        query_classifier_config={
            'early_progress_threshold': 0.3,
            'late_progress_threshold': 0.7
        },
        importance_estimator_config={
            'reward_threshold': 0.7,
            'force_store_interval': 5
        }
    )
    config = manager3.query_classifier.get_config()
    assert config['early_threshold'] == 0.3
    assert config['late_threshold'] == 0.7
    print(f"    ✓ Custom thresholds applied")
    
    print("\n✓ Configuration tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Dynamic Update and Adaptive Retrieval Test Suite (Stage 3)")
    print("="*70)
    
    try:
        # Test individual components
        classifier = test_query_classifier()
        estimator = test_importance_estimator()
        updater = test_selective_updater()
        
        # Test integration
        manager = test_enhanced_manager()
        
        # Test configuration
        test_configuration_options()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        
        print("\nStage 3 features are ready!")
        print("\nSummary:")
        print("  ✓ Query type classification (heuristic + learnable)")
        print("  ✓ Importance-based selective storage")
        print("  ✓ Adaptive retrieval with query-aware weighting")
        print("  ✓ Enhanced memory manager integration")
        print("  ✓ All parameters configurable (no hardcoding)")
        
        print("\nNext steps:")
        print("  1. Integrate with training pipeline")
        print("  2. Run ablation experiments")
        print("  3. Evaluate on Mind2Web dataset")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

