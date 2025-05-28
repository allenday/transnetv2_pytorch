import pytest
import tempfile
import os
import json
import pandas as pd
from transnetv2_pytorch.cli import (
    detect_best_device, 
    get_device, 
    frame_to_timestamp, 
    save_results,
    get_video_fps
)
import torch
from unittest.mock import patch, MagicMock

def get_test_device():
    """Get the device to use for testing based on environment variable"""
    device_env = os.environ.get('TRANSNET_TEST_DEVICE', 'cpu')
    if device_env == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    elif device_env == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class TestUtilityFunctions:
    """Test utility functions in the CLI module"""
    
    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for tests"""
        self.original_env = os.environ.copy()
        os.environ['TRANSNET_TEST_DEVICE'] = get_test_device()
        yield
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_detect_best_device(self):
        """Test automatic device detection"""
        device = detect_best_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_get_device_auto(self):
        """Test get_device with 'auto' argument"""
        device = get_device('auto')
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_get_device_explicit(self):
        """Test get_device with explicit device"""
        device = get_device('cpu')
        assert isinstance(device, torch.device)
        assert device.type == 'cpu'
        
        # Test with cuda if available
        if torch.cuda.is_available():
            device = get_device('cuda')
            assert device.type == 'cuda'
    
    @patch('transnetv2_pytorch.transnetv2_pytorch.TransNetV2.get_video_fps')
    def test_get_video_fps_deprecated(self, mock_get_fps):
        """Test the deprecated get_video_fps function"""
        mock_get_fps.return_value = 30.0
        
        fps = get_video_fps('dummy_video.mp4')
        assert fps == 30.0
        mock_get_fps.assert_called_once_with('dummy_video.mp4')
    
    def test_frame_to_timestamp_deprecated(self):
        """Test the deprecated frame_to_timestamp function"""
        timestamp = frame_to_timestamp(100, 25.0)
        assert timestamp == "4.000"
        
        timestamp = frame_to_timestamp(0, 25.0)
        assert timestamp == "0.000"
        
        timestamp = frame_to_timestamp(150, 29.97)
        expected = 150 / 29.97
        assert timestamp == f"{expected:.3f}"
    
    def test_save_results_csv(self):
        """Test saving results in CSV format"""
        test_data = [
            {'shot_id': 1, 'start_frame': 0, 'end_frame': 50, 'probability': 0.8},
            {'shot_id': 2, 'start_frame': 51, 'end_frame': 100, 'probability': 0.9}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.csv')
            save_results(test_data, output_path, 'csv')
            
            assert os.path.exists(output_path)
            
            # Verify content
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert 'shot_id' in df.columns
            assert df.iloc[0]['shot_id'] == 1
            assert df.iloc[1]['shot_id'] == 2
    
    def test_save_results_json(self):
        """Test saving results in JSON format"""
        test_data = [
            {'shot_id': 1, 'start_frame': 0, 'end_frame': 50, 'probability': 0.8},
            {'shot_id': 2, 'start_frame': 51, 'end_frame': 100, 'probability': 0.9}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.json')
            save_results(test_data, output_path, 'json')
            
            assert os.path.exists(output_path)
            
            # Verify content
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 2
            assert data[0]['shot_id'] == 1
            assert data[1]['shot_id'] == 2
    
    def test_save_results_jsonl(self):
        """Test saving results in JSONL format"""
        test_data = [
            {'shot_id': 1, 'start_frame': 0, 'end_frame': 50, 'probability': 0.8},
            {'shot_id': 2, 'start_frame': 51, 'end_frame': 100, 'probability': 0.9}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.jsonl')
            save_results(test_data, output_path, 'jsonl')
            
            assert os.path.exists(output_path)
            
            # Verify content
            with open(output_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            # Each line should be valid JSON
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            
            assert data1['shot_id'] == 1
            assert data2['shot_id'] == 2
    
    def test_save_results_unsupported_format(self):
        """Test saving results with unsupported format"""
        test_data = [{'shot_id': 1, 'start_frame': 0, 'end_frame': 50}]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.xml')
            
            with pytest.raises(ValueError, match="Unsupported format"):
                save_results(test_data, output_path, 'xml')

if __name__ == '__main__':
    pytest.main([__file__]) 
