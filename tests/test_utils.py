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

class TestUtilityFunctions:
    """Test utility functions from CLI"""
    
    def test_detect_best_device(self):
        """Test device detection"""
        device = detect_best_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_get_device_auto(self):
        """Test get_device with auto option"""
        device = get_device('auto')
        assert isinstance(device, torch.device)
    
    def test_get_device_cpu(self):
        """Test get_device with CPU"""
        device = get_device('cpu')
        assert device == torch.device('cpu')
    
    def test_get_device_invalid(self):
        """Test get_device with invalid device"""
        device = get_device('invalid_device')
        assert isinstance(device, torch.device)  # Should fallback to auto-detection
    
    def test_frame_to_timestamp(self):
        """Test frame to timestamp conversion"""
        # Test with 25 FPS
        timestamp = frame_to_timestamp(25, 25.0)
        assert timestamp == "1.000"
        
        # Test with 30 FPS
        timestamp = frame_to_timestamp(30, 30.0)
        assert timestamp == "1.000"
        
        # Test with fractional result
        timestamp = frame_to_timestamp(15, 30.0)
        assert timestamp == "0.500"
        
        # Test with zero frame
        timestamp = frame_to_timestamp(0, 25.0)
        assert timestamp == "0.000"
    
    def test_save_results_csv(self):
        """Test saving results in CSV format"""
        test_data = [
            {'shot_id': 1, 'start_frame': 0, 'end_frame': 10, 'probability': 0.8},
            {'shot_id': 2, 'start_frame': 11, 'end_frame': 20, 'probability': 0.9}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_results(test_data, temp_path, 'csv')
            
            # Read back and verify
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ['shot_id', 'start_frame', 'end_frame', 'probability']
            assert df.iloc[0]['shot_id'] == 1
            assert df.iloc[1]['shot_id'] == 2
        finally:
            os.unlink(temp_path)
    
    def test_save_results_json(self):
        """Test saving results in JSON format"""
        test_data = [
            {'shot_id': 1, 'start_frame': 0, 'end_frame': 10, 'probability': 0.8},
            {'shot_id': 2, 'start_frame': 11, 'end_frame': 20, 'probability': 0.9}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_results(test_data, temp_path, 'json')
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]['shot_id'] == 1
            assert loaded_data[1]['shot_id'] == 2
        finally:
            os.unlink(temp_path)
    
    def test_save_results_jsonl(self):
        """Test saving results in JSONL format"""
        test_data = [
            {'shot_id': 1, 'start_frame': 0, 'end_frame': 10, 'probability': 0.8},
            {'shot_id': 2, 'start_frame': 11, 'end_frame': 20, 'probability': 0.9}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            save_results(test_data, temp_path, 'jsonl')
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            first_item = json.loads(lines[0])
            second_item = json.loads(lines[1])
            assert first_item['shot_id'] == 1
            assert second_item['shot_id'] == 2
        finally:
            os.unlink(temp_path)
    
    def test_save_results_invalid_format(self):
        """Test saving results with invalid format"""
        test_data = [{'shot_id': 1}]
        
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError):
                save_results(test_data, f.name, 'invalid_format')

if __name__ == '__main__':
    pytest.main([__file__]) 