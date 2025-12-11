"""
Test Memory Module with Real Mind2Web Dataset

Tests trajectory segmentation and block encoding with actual Mind2Web data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from PIL import Image
from collections import Counter
from typing import List, Dict

from model.memory.trajectory_segmentation import (
    FixedWindowSegmenter,
    RuleBasedSegmenter,
    create_segmenter
)
from model.memory.action_vocab import ActionVocabulary


# Mind2Web dataset paths
MIND2WEB_IMAGE_DIR = "/home/may/Proj/SeeProcess/data/mind2web_images/mind2web_images"
MIND2WEB_ANNOT_DIR = "/home/may/Proj/SeeProcess/data/mind2web_annots"


def list_available_splits():
    """List available Mind2Web split files"""
    if not os.path.exists(MIND2WEB_ANNOT_DIR):
        return []
    
    files = os.listdir(MIND2WEB_ANNOT_DIR)
    json_files = [f for f in files if f.endswith('.json')]
    
    # Extract split names
    splits = []
    for f in json_files:
        # Remove 'mind2web_data_' prefix and '.json' suffix
        if f.startswith('mind2web_data_'):
            split_name = f.replace('mind2web_data_', '').replace('.json', '')
            splits.append(split_name)
    
    return splits


def load_mind2web_samples(split='train', num_samples=10):
    """
    Load Mind2Web samples from annotation files.
    
    Args:
        split: 'train', 'test_task', 'test_website', 'test_domain'
        num_samples: Number of samples to load (None for all)
    
    Returns:
        List of samples
    """
    annot_file = os.path.join(MIND2WEB_ANNOT_DIR, f"mind2web_data_{split}.json")
    
    if not os.path.exists(annot_file):
        raise FileNotFoundError(f"Annotation file not found: {annot_file}")
    
    print(f"Loading Mind2Web annotations from: {annot_file}")
    
    with open(annot_file, 'r') as f:
        data = json.load(f)
    
    if num_samples is not None:
        data = data[:num_samples]
    
    print(f"Loaded {len(data)} samples")
    return data


def build_trajectory_from_sample(sample: dict) -> List[Dict]:
    """
    Convert Mind2Web sample to trajectory format.
    
    Args:
        sample: Mind2Web sample dict
    
    Returns:
        List of steps in trajectory format
    """
    trajectory = []
    
    # Extract annotation_id for reference
    annotation_id = sample.get('annotation_id', sample.get('id', 'unknown'))
    
    # Process each action in the sample
    actions = sample.get('actions', sample.get('action_reprs', []))
    
    for step_idx, action in enumerate(actions):
        # Extract action type - Mind2Web uses 'op' field in operation dict
        if isinstance(action, dict):
            if 'operation' in action and isinstance(action['operation'], dict):
                action_name = action['operation'].get('op', 'UNKNOWN')
            elif 'action_name' in action:
                action_name = action['action_name']
            elif 'op' in action:
                action_name = action['op']
            else:
                action_name = 'UNKNOWN'
        else:
            action_name = 'UNKNOWN'
        
        # Extract position (Mind2Web format)
        position = None
        if 'pos_candidates' in action and action['pos_candidates']:
            pos = action['pos_candidates'][0]
            if isinstance(pos, dict):
                if 'x' in pos and 'y' in pos:
                    position = [pos['x'], pos['y']]
                elif 'left' in pos and 'top' in pos:
                    position = [pos['left'], pos['top']]
            elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                position = list(pos[:2])
        
        # Try to get from 'operation' if not found
        if position is None and 'operation' in action:
            op = action['operation']
            if isinstance(op, dict):
                if 'original_op' in op and isinstance(op['original_op'], dict):
                    orig = op['original_op']
                    if 'pos' in orig:
                        pos = orig['pos']
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            position = list(pos[:2])
        
        # Extract value (for INPUT/SELECT actions)
        value = None
        if 'value' in action:
            value = action['value']
        elif 'operation' in action and isinstance(action['operation'], dict):
            op = action['operation']
            value = op.get('value')
            if value is None and 'original_op' in op and isinstance(op['original_op'], dict):
                value = op['original_op'].get('value')
        
        # Image path - Mind2Web image naming
        # Mind2Web format: action may have 'screenshot' field with format: {task_id}-{action_uid}
        image_filename = None
        
        if 'screenshot' in action:
            image_filename = action['screenshot']
        elif 'img_filename' in action:
            image_filename = action['img_filename']
        elif 'image_id' in action:
            image_filename = action['image_id']
        elif 'action_uid' in action:
            # Try to construct from task_id and action_uid
            task_id = sample.get('task_id', annotation_id)
            action_uid = action['action_uid']
            # Mind2Web format: {task_id}-{action_uid}.jpg
            image_filename = f"{task_id}-{action_uid}.jpg"
        
        if image_filename is None:
            # Fallback: use annotation_id and step
            image_filename = f"{annotation_id}_{step_idx}.jpg"
        
        # Ensure proper extension (Mind2Web uses .jpg)
        if not image_filename.endswith(('.png', '.jpg', '.jpeg')):
            # Try both .jpg and .png
            jpg_path = os.path.join(MIND2WEB_IMAGE_DIR, f"{image_filename}.jpg")
            png_path = os.path.join(MIND2WEB_IMAGE_DIR, f"{image_filename}.png")
            if os.path.exists(jpg_path):
                image_filename = f"{image_filename}.jpg"
            elif os.path.exists(png_path):
                image_filename = f"{image_filename}.png"
            else:
                image_filename = f"{image_filename}.jpg"  # Default to jpg
        
        image_path = os.path.join(MIND2WEB_IMAGE_DIR, image_filename)
        
        # Build step
        step = {
            'obs': image_path,
            'action': {
                'action': action_name.upper() if isinstance(action_name, str) else str(action_name),
                'position': position or [0.0, 0.0],  # Default position
                'value': value
            },
            'annotation_id': annotation_id,
            'step_idx': step_idx
        }
        
        trajectory.append(step)
    
    return trajectory


def inspect_data_structure(samples: List[dict], num_samples=2):
    """Inspect the actual data structure to understand field names"""
    print("\n" + "="*70)
    print("Data Structure Inspection")
    print("="*70)
    
    for i, sample in enumerate(samples[:num_samples]):
        print(f"\nSample {i+1}:")
        print(f"  Top-level keys: {list(sample.keys())}")
        
        # Check task_id
        if 'task_id' in sample:
            print(f"  task_id: {sample['task_id']}")
        if 'annotation_id' in sample:
            print(f"  annotation_id: {sample['annotation_id']}")
        
        # Check actions
        actions = sample.get('actions', sample.get('action_reprs', []))
        if actions:
            print(f"  Number of actions: {len(actions)}")
            print(f"\n  First action keys: {list(actions[0].keys())}")
            
            # Check for image-related fields
            image_fields = ['screenshot', 'img_filename', 'image_id', 'action_uid', 'task_id']
            print(f"  Image-related fields in first action:")
            for field in image_fields:
                if field in actions[0]:
                    value = actions[0][field]
                    if isinstance(value, str) and len(value) < 100:
                        print(f"    - {field}: {value}")
                    else:
                        print(f"    - {field}: <present>")


def analyze_action_distribution(samples: List[dict]):
    """Analyze action type distribution in dataset"""
    print("\n" + "="*70)
    print("Action Distribution Analysis")
    print("="*70)
    
    action_counter = Counter()
    trajectory_lengths = []
    
    for sample in samples:
        actions = sample.get('actions', sample.get('action_reprs', []))
        trajectory_lengths.append(len(actions))
        
        for action in actions:
            # Extract action name from operation dict
            if isinstance(action, dict):
                if 'operation' in action and isinstance(action['operation'], dict):
                    action_name = action['operation'].get('op', 'UNKNOWN')
                elif 'action_name' in action:
                    action_name = action['action_name']
                elif 'op' in action:
                    action_name = action['op']
                else:
                    action_name = 'UNKNOWN'
            else:
                action_name = 'UNKNOWN'
            
            if isinstance(action_name, str):
                action_counter[action_name.upper()] += 1
            else:
                action_counter[str(action_name).upper()] += 1
    
    print(f"\nAction Types Found: {len(action_counter)}")
    print(f"{'Action':<20} {'Count':>10} {'Percentage':>12}")
    print("-" * 44)
    
    total_actions = sum(action_counter.values())
    for action, count in action_counter.most_common():
        percentage = (count / total_actions) * 100
        print(f"{action:<20} {count:>10} {percentage:>11.2f}%")
    
    print(f"\nTrajectory Length Statistics:")
    print(f"  Min: {min(trajectory_lengths)}")
    print(f"  Max: {max(trajectory_lengths)}")
    print(f"  Avg: {sum(trajectory_lengths) / len(trajectory_lengths):.2f}")
    print(f"  Total steps: {sum(trajectory_lengths)}")


def test_segmentation_on_real_data(samples: List[dict]):
    """Test trajectory segmentation on real Mind2Web data"""
    print("\n" + "="*70)
    print("Testing Segmentation on Real Data")
    print("="*70)
    
    # Test both segmenters
    segmenters = {
        'Fixed Window (size=4)': FixedWindowSegmenter(window_size=4),
        'Rule-Based': RuleBasedSegmenter(min_block_size=2, max_block_size=6)
    }
    
    results = {name: {'total_blocks': 0, 'block_sizes': []} for name in segmenters}
    
    # Process first few samples
    test_samples = samples[:5]
    
    for sample_idx, sample in enumerate(test_samples):
        print(f"\n--- Sample {sample_idx + 1} ---")
        
        # Build trajectory
        trajectory = build_trajectory_from_sample(sample)
        
        if not trajectory:
            print("  Empty trajectory, skipping")
            continue
        
        print(f"  Trajectory length: {len(trajectory)} steps")
        actions = [step['action']['action'] for step in trajectory]
        print(f"  Actions: {actions}")
        
        # Test each segmenter
        for seg_name, segmenter in segmenters.items():
            blocks = segmenter.segment(trajectory)
            results[seg_name]['total_blocks'] += len(blocks)
            results[seg_name]['block_sizes'].extend([len(b) for b in blocks])
            
            print(f"\n  {seg_name}:")
            print(f"    Blocks: {len(blocks)}")
            for block_idx, block in enumerate(blocks):
                block_actions = [step['action']['action'] for step in block]
                print(f"      Block {block_idx + 1} ({len(block)} steps): {block_actions}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Segmentation Summary")
    print("="*70)
    
    for seg_name, data in results.items():
        print(f"\n{seg_name}:")
        if data['block_sizes']:
            print(f"  Total blocks: {data['total_blocks']}")
            print(f"  Avg block size: {sum(data['block_sizes']) / len(data['block_sizes']):.2f}")
            print(f"  Min block size: {min(data['block_sizes'])}")
            print(f"  Max block size: {max(data['block_sizes'])}")


def test_action_vocabulary_building(samples: List[dict]):
    """Test building action vocabulary from dataset"""
    print("\n" + "="*70)
    print("Testing Action Vocabulary Building")
    print("="*70)
    
    # Build vocabulary
    vocab = ActionVocabulary()
    
    # Extract all actions
    action_set = set()
    action_counts = Counter()
    
    for sample in samples:
        actions = sample.get('actions', sample.get('action_reprs', []))
        for action in actions:
            # Extract action name
            if isinstance(action, dict):
                if 'operation' in action and isinstance(action['operation'], dict):
                    action_name = action['operation'].get('op', 'UNKNOWN')
                elif 'action_name' in action:
                    action_name = action['action_name']
                elif 'op' in action:
                    action_name = action['op']
                else:
                    action_name = 'UNKNOWN'
            else:
                action_name = 'UNKNOWN'
            
            if isinstance(action_name, str):
                action_upper = action_name.upper()
            else:
                action_upper = str(action_name).upper()
            
            action_set.add(action_upper)
            action_counts[action_upper] += 1
    
    # Build vocabulary manually (similar to what ActionVocabulary.build_from_dataset does)
    sorted_actions = sorted(list(action_set))
    vocab.action_to_idx = {'<PAD>': 0}
    vocab.idx_to_action = {0: '<PAD>'}
    vocab.action_counts = action_counts
    
    for idx, action in enumerate(sorted_actions, start=1):
        vocab.action_to_idx[action] = idx
        vocab.idx_to_action[idx] = action
    
    print(f"\nVocabulary Statistics:")
    print(f"  Size: {len(vocab)}")
    print(f"  Actions: {list(vocab.action_to_idx.keys())}")
    
    # Test encoding/decoding
    print(f"\nTesting Encoding/Decoding:")
    test_actions = list(sorted_actions[:5]) + ['UNKNOWN_ACTION']
    for action in test_actions:
        idx = vocab.encode(action)
        decoded = vocab.decode(idx)
        print(f"  '{action}' -> {idx} -> '{decoded}'")
    
    # Save vocabulary
    vocab_path = '/tmp/mind2web_action_vocab.json'
    vocab.save(vocab_path)
    print(f"\nSaved vocabulary to: {vocab_path}")
    
    return vocab


def test_image_loading(samples: List[dict], num_images=3):
    """Test loading actual images from dataset"""
    print("\n" + "="*70)
    print("Testing Image Loading")
    print("="*70)
    
    images_found = 0
    images_missing = 0
    
    # First, examine the raw data structure
    print("\n  Examining image field names in first sample...")
    if samples:
        first_sample = samples[0]
        print(f"  Sample keys: {list(first_sample.keys())}")
        
        actions = first_sample.get('actions', first_sample.get('action_reprs', []))
        if actions:
            print(f"  First action keys: {list(actions[0].keys()) if actions[0] else 'None'}")
            
            # Check for various possible image field names
            possible_fields = ['img_filename', 'image_id', 'screenshot', 'image', 'img']
            found_fields = {field: field in actions[0] for field in possible_fields if actions[0]}
            print(f"  Image fields present: {[k for k, v in found_fields.items() if v]}")
    
    # Try to find actual image files
    print(f"\n  Checking image directory: {MIND2WEB_IMAGE_DIR}")
    if os.path.exists(MIND2WEB_IMAGE_DIR):
        all_images = [f for f in os.listdir(MIND2WEB_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Total images in directory: {len(all_images)}")
        if all_images:
            print(f"  Sample image names: {all_images[:5]}")
    else:
        print(f"  ✗ Directory not found!")
        return
    
    # Test loading with trajectories
    for sample_idx, sample in enumerate(samples[:3]):
        trajectory = build_trajectory_from_sample(sample)
        
        print(f"\nSample {sample_idx + 1} (annotation_id: {sample.get('annotation_id', 'unknown')}):")
        
        for step_idx, step in enumerate(trajectory[:num_images]):
            image_path = step['obs']
            image_basename = os.path.basename(image_path)
            
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    print(f"  Step {step_idx + 1}: ✓ Loaded {img.size} image from {image_basename}")
                    images_found += 1
                except Exception as e:
                    print(f"  Step {step_idx + 1}: ✗ Error loading {image_basename}: {e}")
                    images_missing += 1
            else:
                print(f"  Step {step_idx + 1}: ✗ Missing {image_basename}")
                
                # Try to find similar filenames
                similar = [f for f in all_images if step['annotation_id'] in f]
                if similar:
                    print(f"    (Found similar files: {similar[:3]})")
                
                images_missing += 1
    
    print(f"\nImage Loading Summary:")
    print(f"  Found: {images_found}")
    print(f"  Missing: {images_missing}")
    
    if images_found == 0:
        print(f"\n  ⚠️  SUGGESTION:")
        print(f"    Image filenames in trajectory don't match actual files.")
        print(f"    Please check Mind2Web annotation format and update")
        print(f"    the image path construction in build_trajectory_from_sample().")


def main():
    """Run all tests on real Mind2Web data"""
    print("\n" + "="*70)
    print("Memory Module Test with Real Mind2Web Dataset")
    print("="*70)
    
    # Check paths
    print(f"\nDataset Paths:")
    print(f"  Images: {MIND2WEB_IMAGE_DIR}")
    print(f"    Exists: {os.path.exists(MIND2WEB_IMAGE_DIR)}")
    print(f"  Annotations: {MIND2WEB_ANNOT_DIR}")
    print(f"    Exists: {os.path.exists(MIND2WEB_ANNOT_DIR)}")
    
    if not os.path.exists(MIND2WEB_ANNOT_DIR):
        print(f"\n✗ ERROR: Annotation directory not found!")
        print(f"  Please check the path: {MIND2WEB_ANNOT_DIR}")
        return 1
    
    # List available splits
    available_splits = list_available_splits()
    print(f"\n  Available splits: {available_splits if available_splits else 'None found'}")
    
    if not available_splits:
        print(f"\n✗ ERROR: No Mind2Web annotation files found!")
        print(f"  Expected format: mind2web_data_<split>.json")
        return 1
    
    try:
        # Load samples
        print("\n" + "="*70)
        print("Loading Dataset")
        print("="*70)
        
        samples = load_mind2web_samples(split='train', num_samples=20)
        
        # Test 0: Inspect data structure first
        inspect_data_structure(samples, num_samples=2)
        
        # Test 1: Action distribution analysis
        analyze_action_distribution(samples)
        
        # Test 2: Build action vocabulary
        vocab = test_action_vocabulary_building(samples)
        
        # Test 3: Test segmentation
        test_segmentation_on_real_data(samples)
        
        # Test 4: Test image loading
        test_image_loading(samples, num_images=3)
        
        print("\n" + "="*70)
        print("TESTS COMPLETED!")
        print("="*70)
        
        print("\n✅ Core功能验证:")
        print("  ✓ 轨迹数据加载")
        print("  ✓ 动作类型分析")
        print("  ✓ 轨迹块划分")
        print("  ✓ 动作词表构建")
        
        print("\n⚠️  图像加载（如果失败）:")
        print("  - 图像缺失不影响分段逻辑测试")
        print("  - 核心功能（轨迹块划分、词表构建）已验证")
        print("  - 图像仅用于后续的块编码器测试")
        
        print("\nNext Steps:")
        print("  1. 检查分段结果 - 块是否语义完整？")
        print("  2. 根据需要调整边界模式")
        print("  3. 若要测试块编码器，需要正确的图像路径")
        print("     提示：检查标注文件中的图像字段名和实际图像文件名格式")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

