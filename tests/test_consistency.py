#!/usr/bin/env python3
"""
Unit test to reproduce the TransNetV2 consistency bug through multiple build/destroy cycles.

The bug: detect_scenes() returns different results (195 -> 22 -> 1 scenes) when called
in different contexts, suggesting state retention or caching issues.
"""

import unittest
import os
import sys
import gc
import torch
from transnetv2_pytorch import TransNetV2


class TestTransNetV2Consistency(unittest.TestCase):
    """Test suite to reproduce and verify the consistency bug"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_video = None
        # Find a test video
        for candidate in ["tests/test.mp4", "tests/fixtures/media/music-video.mp4"]:
            if os.path.exists(candidate):
                cls.test_video = candidate
                break
        
        if cls.test_video is None:
            raise unittest.SkipTest("No test video found")
        
        cls.threshold = 0.5
        cls.device = 'cpu'  # Use CPU to avoid MPS complications
        
    def tearDown(self):
        """Clean up after each test"""
        # Force cleanup between tests
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    def test_single_model_multiple_calls(self):
        """Test: Same model instance, multiple detect_scenes calls should be consistent"""
        print(f"\n=== Test 1: Single model, multiple calls ===")
        
        model = TransNetV2(device=self.device)
        results = []
        
        for i in range(3):
            scenes = model.detect_scenes(self.test_video, threshold=self.threshold)
            scene_count = len(scenes)
            results.append(scene_count)
            print(f"  Run {i+1}: {scene_count} scenes")
            
            if i == 0:
                first_scenes = scenes[:3] if len(scenes) >= 3 else scenes
                print(f"  First few scenes: {[(s['start_time'], s['end_time']) for s in first_scenes]}")
        
        print(f"  Results: {results}")
        # All results should be identical
        self.assertEqual(len(set(results)), 1, 
                        f"Single model gave inconsistent results: {results}")
    
    def test_multiple_model_instances_same_process(self):
        """Test: Multiple model instances in same process should give same results"""
        print(f"\n=== Test 2: Multiple models, same process ===")
        
        results = []
        
        for i in range(3):
            # Create fresh model instance each time
            model = TransNetV2(device=self.device)
            scenes = model.detect_scenes(self.test_video, threshold=self.threshold)
            scene_count = len(scenes)
            results.append(scene_count)
            print(f"  Model {i+1}: {scene_count} scenes")
            
            # Explicitly delete model to force cleanup
            del model
            gc.collect()
        
        print(f"  Results: {results}")
        # All results should be identical
        self.assertEqual(len(set(results)), 1, 
                        f"Multiple models gave inconsistent results: {results}")
    
    def test_model_state_isolation(self):
        """Test: Model state should not affect subsequent instances"""
        print(f"\n=== Test 3: Model state isolation ===")
        
        # First model - run prediction
        model1 = TransNetV2(device=self.device)
        scenes1 = model1.detect_scenes(self.test_video, threshold=self.threshold)
        count1 = len(scenes1)
        print(f"  Model 1: {count1} scenes")
        
        # Do some operations that might affect internal state
        _ = model1.get_scene_count(self.test_video, threshold=0.3)  # Different threshold
        _ = model1.get_scene_timestamps(self.test_video, threshold=0.7)  # Different threshold
        
        # Run again with original threshold
        scenes1b = model1.detect_scenes(self.test_video, threshold=self.threshold)
        count1b = len(scenes1b)
        print(f"  Model 1 (after other ops): {count1b} scenes")
        
        # Create second model after first has been used
        model2 = TransNetV2(device=self.device)
        scenes2 = model2.detect_scenes(self.test_video, threshold=self.threshold)
        count2 = len(scenes2)
        print(f"  Model 2 (fresh): {count2} scenes")
        
        # All should be the same
        results = [count1, count1b, count2]
        print(f"  Results: {results}")
        self.assertEqual(len(set(results)), 1, 
                        f"Model state affected subsequent instances: {results}")
    
    def test_threshold_consistency(self):
        """Test: Same threshold should always give same results"""
        print(f"\n=== Test 4: Threshold consistency ===")
        
        model = TransNetV2(device=self.device)
        
        # Test multiple thresholds
        thresholds = [0.3, 0.5, 0.7]
        threshold_results = {}
        
        for threshold in thresholds:
            counts = []
            for i in range(2):
                scenes = model.detect_scenes(self.test_video, threshold=threshold)
                counts.append(len(scenes))
            
            threshold_results[threshold] = counts
            print(f"  Threshold {threshold}: {counts}")
            
            # Same threshold should give same results
            self.assertEqual(len(set(counts)), 1, 
                            f"Threshold {threshold} gave inconsistent results: {counts}")
    
    def test_memory_pressure_consistency(self):
        """Test: Results should be consistent under memory pressure"""
        print(f"\n=== Test 5: Memory pressure consistency ===")
        
        results = []
        
        for i in range(3):
            # Create some memory pressure
            if i > 0:
                # Allocate and release some tensors
                dummy_tensors = [torch.randn(1000, 1000) for _ in range(10)]
                del dummy_tensors
                gc.collect()
            
            model = TransNetV2(device=self.device)
            scenes = model.detect_scenes(self.test_video, threshold=self.threshold)
            scene_count = len(scenes)
            results.append(scene_count)
            print(f"  Run {i+1} (memory pressure): {scene_count} scenes")
            
            del model
            gc.collect()
        
        print(f"  Results: {results}")
        self.assertEqual(len(set(results)), 1, 
                        f"Memory pressure caused inconsistent results: {results}")
    
    def test_wrapper_function_consistency(self):
        """Test: Wrapper functions should not affect results"""
        print(f"\n=== Test 6: Wrapper function consistency ===")
        
        def detect_scenes_wrapper(video_path, threshold):
            """Wrapper function that might affect state"""
            model = TransNetV2(device=self.device)
            return model.detect_scenes(video_path, threshold=threshold)
        
        def detect_scenes_nested_wrapper(video_path, threshold):
            """Nested wrapper function"""
            def inner():
                return detect_scenes_wrapper(video_path, threshold)
            return inner()
        
        # Direct call
        model = TransNetV2(device=self.device)
        direct_scenes = model.detect_scenes(self.test_video, threshold=self.threshold)
        direct_count = len(direct_scenes)
        
        # Wrapper call
        wrapper_scenes = detect_scenes_wrapper(self.test_video, self.threshold)
        wrapper_count = len(wrapper_scenes)
        
        # Nested wrapper call
        nested_scenes = detect_scenes_nested_wrapper(self.test_video, self.threshold)
        nested_count = len(nested_scenes)
        
        results = [direct_count, wrapper_count, nested_count]
        print(f"  Direct: {direct_count}, Wrapper: {wrapper_count}, Nested: {nested_count}")
        
        self.assertEqual(len(set(results)), 1, 
                        f"Wrapper functions caused inconsistent results: {results}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2) 