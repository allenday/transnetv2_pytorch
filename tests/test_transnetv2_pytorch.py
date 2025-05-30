import pytest
import torch
import numpy as np
import os
from unittest.mock import patch, MagicMock
from transnetv2_pytorch import TransNetV2

# Get the path to the test video
TEST_VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'test.mp4')

def get_test_device():
    """Get the device to use for testing based on environment variable"""
    device_env = os.environ.get('TRANSNET_TEST_DEVICE', 'cpu')
    if device_env == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    elif device_env == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

class TestTransNetV2Enhanced:
    """Test the enhanced TransNetV2 API features"""
    
    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for tests"""
        self.original_env = os.environ.copy()
        os.environ['TRANSNET_TEST_DEVICE'] = get_test_device()
        yield
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @pytest.fixture
    def model(self):
        """Create a TransNetV2 model for testing"""
        device = get_test_device()
        return TransNetV2(device=device)
    
    def test_device_auto_detection(self):
        """Test automatic device detection"""
        model = TransNetV2(device='auto')
        assert model.device.type in ['cuda', 'mps', 'cpu']
        
        # Test static method directly
        best_device = TransNetV2._detect_best_device()
        assert best_device in ['cuda', 'mps', 'cpu']
    
    def test_device_explicit_setting(self):
        """Test explicit device setting"""
        model = TransNetV2(device='cpu')
        assert model.device.type == 'cpu'
    
    @patch('ffmpeg.probe')
    def test_get_video_fps_success(self, mock_probe, model):
        """Test successful FPS extraction"""
        # Mock ffmpeg probe response
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'r_frame_rate': '25/1'
                }
            ]
        }
        
        fps = model.get_video_fps('dummy_video.mp4')
        assert fps == 25.0
        
        # Test fractional FPS
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'r_frame_rate': '30000/1001'
                }
            ]
        }
        
        fps = model.get_video_fps('dummy_video.mp4')
        assert abs(fps - 29.97) < 0.01
    
    @patch('ffmpeg.probe')
    def test_get_video_fps_no_video_stream(self, mock_probe, model):
        """Test FPS extraction when no video stream found"""
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'audio'
                }
            ]
        }
        
        fps = model.get_video_fps('dummy_video.mp4')
        assert fps == 25.0  # Default fallback
    
    @patch('ffmpeg.probe')
    def test_get_video_fps_exception(self, mock_probe, model):
        """Test FPS extraction when ffmpeg fails"""
        mock_probe.side_effect = Exception("ffmpeg error")
        
        fps = model.get_video_fps('dummy_video.mp4')
        assert fps == 25.0  # Default fallback
    
    def test_frame_to_timestamp(self):
        """Test frame to timestamp conversion"""
        # Test basic conversion
        timestamp = TransNetV2.frame_to_timestamp(150, 25.0)
        assert timestamp == "6.000"
        
        # Test fractional result
        timestamp = TransNetV2.frame_to_timestamp(100, 29.97)
        expected = 100 / 29.97
        assert timestamp == f"{expected:.3f}"
        
        # Test zero frame
        timestamp = TransNetV2.frame_to_timestamp(0, 25.0)
        assert timestamp == "0.000"
    
    def test_predictions_to_scenes_with_data_basic(self, model):
        """Test basic predictions to scenes with data conversion"""
        # Create test predictions with clear boundaries
        predictions = np.array([0.1, 0.1, 0.8, 0.1, 0.1, 0.9, 0.1, 0.1])
        
        scenes = model.predictions_to_scenes_with_data(predictions, fps=25.0, threshold=0.5)
        
        assert len(scenes) > 0
        
        # Check structure of first scene
        scene = scenes[0]
        required_fields = ['shot_id', 'start_frame', 'end_frame', 'start_time', 'end_time', 'probability']
        for field in required_fields:
            assert field in scene
        
        # Check shot_id starts from 1
        assert scene['shot_id'] == 1
        
        # Check timestamps are present
        assert 'start_time' in scene
        assert 'end_time' in scene
    
    def test_predictions_to_scenes_with_data_no_fps(self, model):
        """Test predictions to scenes without FPS"""
        predictions = np.array([0.1, 0.1, 0.8, 0.1, 0.1])
        
        scenes = model.predictions_to_scenes_with_data(predictions, threshold=0.5)
        
        assert len(scenes) > 0
        scene = scenes[0]
        
        # Should have frame data but no timestamps
        assert 'shot_id' in scene
        assert 'start_frame' in scene
        assert 'end_frame' in scene
        assert 'probability' in scene
        assert 'start_time' not in scene
        assert 'end_time' not in scene
    
    @patch('transnetv2_pytorch.transnetv2_pytorch.TransNetV2.get_video_fps')
    def test_predictions_to_scenes_with_data_video_path(self, mock_get_fps, model):
        """Test predictions to scenes with video path for FPS extraction"""
        mock_get_fps.return_value = 30.0
        
        predictions = np.array([0.1, 0.1, 0.8, 0.1, 0.1])
        
        scenes = model.predictions_to_scenes_with_data(
            predictions, 
            video_path='dummy_video.mp4', 
            threshold=0.5
        )
        
        assert len(scenes) > 0
        scene = scenes[0]
        
        # Should have timestamps since FPS was extracted
        assert 'start_time' in scene
        assert 'end_time' in scene
        
        # Verify FPS extraction was called
        mock_get_fps.assert_called_once_with('dummy_video.mp4')
    
    def test_predictions_to_scenes_with_data_torch_tensor(self, model):
        """Test predictions to scenes with torch tensor input"""
        # Create torch tensor predictions
        predictions = torch.tensor([0.1, 0.1, 0.8, 0.1, 0.1], dtype=torch.float32)
        
        scenes = model.predictions_to_scenes_with_data(predictions, fps=25.0, threshold=0.5)
        
        assert len(scenes) > 0
        scene = scenes[0]
        assert 'shot_id' in scene
        assert isinstance(scene['probability'], float)
    
    @patch('transnetv2_pytorch.transnetv2_pytorch.TransNetV2.predict_video')
    @patch('transnetv2_pytorch.transnetv2_pytorch.TransNetV2.get_video_fps')
    def test_analyze_video(self, mock_get_fps, mock_predict_video, model):
        """Test comprehensive video analysis"""
        # Mock the dependencies
        mock_get_fps.return_value = 25.0
        
        # Create mock video frames and predictions
        mock_video_frames = torch.zeros(100, 27, 48, 3, dtype=torch.uint8)
        mock_single_pred = torch.tensor([0.1] * 50 + [0.8] + [0.1] * 49)
        mock_all_pred = torch.tensor([0.1] * 100)
        
        mock_predict_video.return_value = (mock_video_frames, mock_single_pred, mock_all_pred)
        
        # Test the method
        results = model.analyze_video('dummy_video.mp4', threshold=0.5)
        
        # Verify the structure
        assert 'video_frames' in results
        assert 'single_frame_predictions' in results
        assert 'all_frame_predictions' in results
        assert 'fps' in results
        assert 'scenes' in results
        assert 'total_scenes' in results
        
        # Check FPS
        assert results['fps'] == 25.0
        
        # Check that scenes were created
        scenes = results['scenes']
        assert len(scenes) > 0
        assert results['total_scenes'] == len(scenes)
        
        # Verify scene structure
        scene = scenes[0]
        assert 'shot_id' in scene
        assert 'start_frame' in scene
        assert 'end_frame' in scene
        assert 'probability' in scene
    
    def test_predictions_to_scenes_with_data_probability_calculation(self, model):
        """Test that probability calculation works correctly"""
        # Create predictions where we know the max values
        predictions = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.8, 0.2, 0.1])
        
        scenes = model.predictions_to_scenes_with_data(predictions, fps=25.0, threshold=0.5)
        
        # Should detect scenes at positions 2 and 5
        assert len(scenes) >= 2
        
        # Check that probabilities are calculated correctly
        for scene in scenes:
            assert 0.0 <= scene['probability'] <= 1.0
            assert isinstance(scene['probability'], float)
    
    def test_predictions_to_scenes_with_data_empty_predictions(self, model):
        """Test with predictions that don't trigger any scenes"""
        predictions = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        scenes = model.predictions_to_scenes_with_data(predictions, fps=25.0, threshold=0.5)
        
        # Should still return one scene covering the whole video
        assert len(scenes) == 1
        scene = scenes[0]
        assert scene['start_frame'] == 0
        assert scene['end_frame'] == len(predictions) - 1
    
    def test_predictions_to_scenes_with_data_all_high_predictions(self, model):
        """Test with all high predictions"""
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        
        scenes = model.predictions_to_scenes_with_data(predictions, fps=25.0, threshold=0.5)
        
        # Should handle this edge case gracefully
        assert len(scenes) >= 1
        for scene in scenes:
            assert scene['probability'] > 0.5
    
    def test_enhanced_api_integration(self, model):
        """Test integration of enhanced API features"""
        # Test that all new methods work together
        
        # 1. Device detection
        device = TransNetV2._detect_best_device()
        assert device in ['cuda', 'mps', 'cpu']
        
        # 2. Frame to timestamp
        timestamp = TransNetV2.frame_to_timestamp(100, 25.0)
        assert timestamp == "4.000"
        
        # 3. Predictions to scenes with data
        predictions = np.array([0.1, 0.1, 0.8, 0.1, 0.1])
        scenes = model.predictions_to_scenes_with_data(predictions, fps=25.0)
        
        assert len(scenes) > 0
        assert all('shot_id' in scene for scene in scenes)
        assert all('start_time' in scene for scene in scenes)
        assert all('end_time' in scene for scene in scenes) 