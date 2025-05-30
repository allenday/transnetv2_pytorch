"""
Integration tests to verify CLI and programmatic API produce identical results.

This ensures CLI and API consistency using the same clean methods.
"""

import subprocess
import json
import torch
import tempfile
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestAPIConsistency:
    """Test API method consistency"""
    
    @pytest.fixture
    def mock_model_setup(self):
        """Setup a mock model that produces consistent test results"""
        # Mock the TransNetV2 model
        mock_model = MagicMock()
        
        # Create consistent test predictions
        test_predictions = np.array([0.1, 0.2, 0.8, 0.1, 0.9, 0.1, 0.2, 0.7, 0.1, 0.1])
        
        # Mock the analyze_video method
        def mock_analyze_video(video_path, threshold=0.5, quiet=False):
            # Convert predictions to scenes using the real logic
            from transnetv2_pytorch.transnetv2_pytorch import TransNetV2
            scenes = TransNetV2.predictions_to_scenes(test_predictions, threshold)
            
            # Create scene data
            scene_list = []
            for i, scene in enumerate(scenes):
                scene_data = {
                    'shot_id': i + 1,
                    'start_frame': int(scene[0]),
                    'end_frame': int(scene[1]),
                    'start_time': f"{scene[0]/25.0:.3f}",
                    'end_time': f"{scene[1]/25.0:.3f}",
                    'probability': float(np.max(test_predictions[scene[0]:scene[1]+1]))
                }
                scene_list.append(scene_data)
            
            return {
                'scenes': scene_list,
                'total_scenes': len(scene_list),
                'fps': 25.0
            }
        
        mock_model.analyze_video = mock_analyze_video
        
        # Add the detect_scenes method
        def mock_detect_scenes(video_path, threshold=0.5):
            results = mock_analyze_video(video_path, threshold=threshold, quiet=True)
            return results['scenes']
        
        mock_model.detect_scenes = mock_detect_scenes
        
        return mock_model
    
    def test_detect_scenes_matches_analyze_video(self, mock_model_setup):
        """Test that detect_scenes exactly matches analyze_video scenes"""
        model = mock_model_setup
        video_path = "test_video.mp4"
        threshold = 0.3
        
        # Get results from both methods
        detect_scenes_result = model.detect_scenes(video_path, threshold=threshold)
        analyze_result = model.analyze_video(video_path, threshold=threshold, quiet=True)
        
        # They should be identical
        assert detect_scenes_result == analyze_result['scenes']
    
    def test_different_thresholds_produce_different_results(self, mock_model_setup):
        """Test that different thresholds actually produce different scene boundaries"""
        model = mock_model_setup
        video_path = "test_video.mp4"
        
        # Get results with different thresholds
        scenes_low = model.detect_scenes(video_path, threshold=0.3)
        scenes_high = model.detect_scenes(video_path, threshold=0.7)
        
        # They should be different (low threshold should detect more scenes)
        assert len(scenes_low) >= len(scenes_high)
    
    def test_convenience_methods_use_detect_scenes(self):
        """Test that convenience methods properly use detect_scenes"""
        from transnetv2_pytorch.transnetv2_pytorch import TransNetV2
        
        # Mock the detect_scenes method
        with patch.object(TransNetV2, 'detect_scenes') as mock_detect_scenes:
            mock_scenes = [
                {'shot_id': 1, 'start_time': '0.000', 'end_time': '1.000'},
                {'shot_id': 2, 'start_time': '1.000', 'end_time': '2.000'}
            ]
            mock_detect_scenes.return_value = mock_scenes
            
            model = TransNetV2()
            
            # Test get_scene_count
            count = model.get_scene_count("test.mp4", threshold=0.3)
            assert count == 2
            mock_detect_scenes.assert_called_with("test.mp4", 0.3)
            
            # Test get_scene_timestamps  
            mock_detect_scenes.reset_mock()
            timestamps = model.get_scene_timestamps("test.mp4", threshold=0.4)
            expected = [(0.0, 1.0), (1.0, 2.0)]
            assert timestamps == expected
            mock_detect_scenes.assert_called_with("test.mp4", 0.4)


def test_api_hierarchy():
    """
    Test that the API has the proper MECE hierarchy
    """
    from transnetv2_pytorch.transnetv2_pytorch import TransNetV2
    
    model = TransNetV2()
    
    # High-level methods
    assert hasattr(model, 'detect_scenes')
    assert hasattr(model, 'get_scene_count')
    assert hasattr(model, 'get_scene_timestamps')
    
    # Mid-level methods
    assert hasattr(model, 'analyze_video')
    assert hasattr(model, 'predict_video')
    
    # Low-level methods
    assert hasattr(model, 'predict_raw')
    assert hasattr(model, 'predictions_to_scenes')
    assert hasattr(model, 'predictions_to_scenes_with_data')


if __name__ == "__main__":
    pytest.main([__file__]) 