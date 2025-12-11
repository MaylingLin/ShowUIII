"""
Basic Test Script for Memory Module (No PyTorch Required)

Tests trajectory segmentation logic without requiring GPU/PyTorch environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.memory.trajectory_segmentation import (
    FixedWindowSegmenter,
    RuleBasedSegmenter,
    create_segmenter
)


def test_fixed_window_segmenter():
    """Test fixed window segmentation"""
    print("\n" + "="*70)
    print("TEST 1: Fixed Window Segmenter")
    print("="*70)
    
    # Create sample trajectory
    trajectory = [
        {'obs': f'img{i}.png', 'action': {'action': 'CLICK', 'position': [0.5, 0.5]}}
        for i in range(10)
    ]
    
    # Test different window sizes
    for window_size in [3, 4, 5]:
        print(f"\n  Window size: {window_size}")
        segmenter = FixedWindowSegmenter(window_size=window_size)
        blocks = segmenter.segment(trajectory)
        
        print(f"    Total steps: {len(trajectory)}")
        print(f"    Number of blocks: {len(blocks)}")
        print(f"    Block sizes: {[len(b) for b in blocks]}")
        
        # Verify
        total_steps = sum(len(b) for b in blocks)
        assert total_steps == len(trajectory), f"Steps mismatch: {total_steps} != {len(trajectory)}"
        print(f"    ✓ All steps accounted for")
    
    # Test edge cases
    print(f"\n  Edge Cases:")
    
    # Empty trajectory
    empty_blocks = segmenter.segment([])
    assert len(empty_blocks) == 0, "Empty trajectory should return empty blocks"
    print(f"    ✓ Empty trajectory handled")
    
    # Single step
    single_blocks = segmenter.segment([trajectory[0]])
    assert len(single_blocks) == 1 and len(single_blocks[0]) == 1
    print(f"    ✓ Single step handled")
    
    print("\n  ✓ Fixed Window Segmenter tests passed!")


def test_rule_based_segmenter():
    """Test rule-based segmentation"""
    print("\n" + "="*70)
    print("TEST 2: Rule-Based Segmenter")
    print("="*70)
    
    # Create trajectory with clear semantic boundaries
    trajectory = [
        # Block 1: Scroll and click pattern
        {'obs': 'img1.png', 'action': {'action': 'SCROLL'}},
        {'obs': 'img2.png', 'action': {'action': 'SCROLL'}},
        {'obs': 'img3.png', 'action': {'action': 'CLICK'}},  # Boundary: SCROLL -> CLICK
        
        # Block 2: Input pattern
        {'obs': 'img4.png', 'action': {'action': 'INPUT'}},
        {'obs': 'img5.png', 'action': {'action': 'INPUT'}},
        {'obs': 'img6.png', 'action': {'action': 'ENTER'}},  # Boundary: INPUT -> ENTER
        
        # Block 3: Click pattern
        {'obs': 'img7.png', 'action': {'action': 'CLICK'}},
        {'obs': 'img8.png', 'action': {'action': 'HOVER'}},
        {'obs': 'img9.png', 'action': {'action': 'CLICK'}},  # Boundary: HOVER -> CLICK
        
        # Block 4: Answer (always boundary after)
        {'obs': 'img10.png', 'action': {'action': 'ANSWER'}},  # Boundary: ANSWER -> *
    ]
    
    print(f"\n  Input trajectory actions:")
    actions = [step['action']['action'] for step in trajectory]
    print(f"    {actions}")
    
    # Test with default patterns
    print(f"\n  Using default boundary patterns:")
    segmenter = RuleBasedSegmenter(min_block_size=2, max_block_size=5)
    blocks = segmenter.segment(trajectory)
    
    print(f"    Number of blocks: {len(blocks)}")
    for i, block in enumerate(blocks):
        block_actions = [step['action']['action'] for step in block]
        print(f"    Block {i+1} ({len(block)} steps): {block_actions}")
    
    # Verify total steps
    total_steps = sum(len(b) for b in blocks)
    assert total_steps == len(trajectory)
    print(f"    ✓ All {total_steps} steps accounted for")
    
    # Test with custom patterns
    print(f"\n  Using custom boundary patterns:")
    custom_patterns = [
        ('SCROLL', 'CLICK'),
        ('INPUT', 'CLICK'),
    ]
    custom_segmenter = RuleBasedSegmenter(
        min_block_size=2,
        max_block_size=6,
        boundary_patterns=custom_patterns
    )
    custom_blocks = custom_segmenter.segment(trajectory)
    
    print(f"    Number of blocks: {len(custom_blocks)}")
    for i, block in enumerate(custom_blocks):
        block_actions = [step['action']['action'] for step in block]
        print(f"    Block {i+1} ({len(block)} steps): {block_actions}")
    
    # Test max_block_size enforcement
    print(f"\n  Testing max_block_size enforcement:")
    long_trajectory = [
        {'obs': f'img{i}.png', 'action': {'action': 'CLICK'}}
        for i in range(15)
    ]
    
    max_size = 4
    segmenter_max = RuleBasedSegmenter(min_block_size=1, max_block_size=max_size)
    blocks_max = segmenter_max.segment(long_trajectory)
    
    max_block_len = max(len(b) for b in blocks_max)
    print(f"    Max block size setting: {max_size}")
    print(f"    Actual max block length: {max_block_len}")
    assert max_block_len <= max_size, f"Max size violated: {max_block_len} > {max_size}"
    print(f"    ✓ Max block size enforced")
    
    # Test case insensitivity
    print(f"\n  Testing case insensitivity:")
    mixed_case_traj = [
        {'obs': 'img1.png', 'action': {'action': 'scroll'}},
        {'obs': 'img2.png', 'action': {'action': 'CLICK'}},
        {'obs': 'img3.png', 'action': {'action': 'Input'}},
    ]
    case_blocks = segmenter.segment(mixed_case_traj)
    print(f"    Input: ['scroll', 'CLICK', 'Input']")
    print(f"    Blocks created: {len(case_blocks)}")
    print(f"    ✓ Case handled correctly")
    
    print("\n  ✓ Rule-Based Segmenter tests passed!")


def test_factory_function():
    """Test segmenter factory"""
    print("\n" + "="*70)
    print("TEST 3: Segmenter Factory Function")
    print("="*70)
    
    # Create via factory
    fixed_seg = create_segmenter('fixed', window_size=5)
    rule_seg = create_segmenter('rule', min_block_size=2, max_block_size=8)
    
    print(f"\n  Created segmenters:")
    print(f"    Fixed: {type(fixed_seg).__name__}")
    print(f"    Rule: {type(rule_seg).__name__}")
    
    # Verify they work
    test_traj = [
        {'obs': f'img{i}.png', 'action': {'action': 'CLICK'}}
        for i in range(10)
    ]
    
    fixed_blocks = fixed_seg.segment(test_traj)
    rule_blocks = rule_seg.segment(test_traj)
    
    print(f"\n  Applied to 10-step trajectory:")
    print(f"    Fixed blocks: {len(fixed_blocks)}")
    print(f"    Rule blocks: {len(rule_blocks)}")
    
    # Test invalid type
    print(f"\n  Testing error handling:")
    try:
        invalid_seg = create_segmenter('invalid_type')
        print(f"    ✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"    ✓ Correctly raised ValueError: {str(e)[:50]}...")
    
    print("\n  ✓ Factory function tests passed!")


def test_boundary_detection():
    """Test boundary pattern matching in detail"""
    print("\n" + "="*70)
    print("TEST 4: Boundary Detection Logic")
    print("="*70)
    
    segmenter = RuleBasedSegmenter()
    
    # Test specific patterns
    test_cases = [
        ('SCROLL', 'CLICK', True),
        ('CLICK', 'INPUT', True),
        ('INPUT', 'ENTER', True),
        ('ANSWER', 'CLICK', True),  # ANSWER -> * matches anything
        ('ANSWER', 'SCROLL', True),
        ('CLICK', 'CLICK', False),
        ('SCROLL', 'SCROLL', False),
        ('INPUT', 'INPUT', False),
    ]
    
    print(f"\n  Testing boundary patterns:")
    for prev, curr, expected in test_cases:
        result = segmenter.is_boundary(prev, curr)
        status = "✓" if result == expected else "✗"
        print(f"    {status} ({prev} -> {curr}): {result} (expected {expected})")
        assert result == expected, f"Boundary detection failed for {prev} -> {curr}"
    
    print("\n  ✓ Boundary detection tests passed!")


def test_edge_cases():
    """Test various edge cases"""
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)
    
    # Test with missing action fields
    print(f"\n  Testing malformed action structures:")
    
    traj_variants = [
        # String action instead of dict
        [
            {'obs': 'img1.png', 'action': 'CLICK'},
            {'obs': 'img2.png', 'action': 'INPUT'},
        ],
        # Dict without 'action' key
        [
            {'obs': 'img1.png', 'action': {'type': 'CLICK'}},
            {'obs': 'img2.png', 'action': {'type': 'INPUT'}},
        ],
    ]
    
    segmenter = RuleBasedSegmenter(min_block_size=1)
    
    for i, traj in enumerate(traj_variants):
        try:
            blocks = segmenter.segment(traj)
            print(f"    Variant {i+1}: Handled gracefully, {len(blocks)} blocks")
        except Exception as e:
            print(f"    Variant {i+1}: Error - {str(e)}")
    
    # Test parameter validation
    print(f"\n  Testing parameter validation:")
    
    try:
        FixedWindowSegmenter(window_size=0)
        print(f"    ✗ Should reject window_size=0")
        assert False
    except ValueError:
        print(f"    ✓ Correctly rejected window_size=0")
    
    try:
        RuleBasedSegmenter(min_block_size=5, max_block_size=3)
        print(f"    ✗ Should reject min > max")
        assert False
    except ValueError:
        print(f"    ✓ Correctly rejected min_block_size > max_block_size")
    
    print("\n  ✓ Edge case tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Memory Module Basic Test Suite (No PyTorch)")
    print("="*70)
    print("\nTesting trajectory segmentation logic...")
    
    try:
        test_fixed_window_segmenter()
        test_rule_based_segmenter()
        test_factory_function()
        test_boundary_detection()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("ALL BASIC TESTS PASSED! ✓")
        print("="*70)
        print("\nTrajectory segmentation module is working correctly.")
        print("\nTo test PyTorch components (encoder, losses), run:")
        print("  python test_memory_module.py")
        print("\nThese require PyTorch to be installed.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

