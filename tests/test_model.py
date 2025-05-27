import pytest
import torch
import numpy as np
import os
from transnetv2_pytorch import TransNetV2

# Get the path to the test video
TEST_VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'test.mp4')

def get_test_device():
    """Get the device to use for testing based on environment variable"""
    device_env = os.environ.get('TRANSNET_TEST_DEVICE', 'cpu')
    if device_env == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif device_env == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class TestTransNetV2Model:
    """Test the TransNetV2 model functionality"""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing"""
        device = get_test_device()
        model = TransNetV2(device=device)
        return model
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly"""
        assert isinstance(model, TransNetV2)
        assert model.device == torch.device('cpu')
    
    def test_model_forward_pass(self, model):
        """Test model forward pass with dummy data"""
        # Create dummy input: batch_size=1, frames=100, height=27, width=48, channels=3
        dummy_input = torch.randint(0, 256, (1, 100, 27, 48, 3), dtype=torch.uint8)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output format
        if isinstance(output, tuple):
            single_frame_pred, all_frames_pred = output
            assert single_frame_pred.shape == (1, 100, 1)
            assert 'many_hot' in all_frames_pred
            assert all_frames_pred['many_hot'].shape == (1, 100, 1)
        else:
            assert output.shape == (1, 100, 1)
    
    def test_predict_raw(self, model):
        """Test predict_raw method"""
        # Create dummy input
        dummy_input = torch.randint(0, 256, (1, 100, 27, 48, 3), dtype=torch.uint8)
        
        with torch.no_grad():
            single_frame_pred, all_frames_pred = model.predict_raw(dummy_input)
        
        # Check that outputs are probabilities (between 0 and 1)
        assert torch.all(single_frame_pred >= 0) and torch.all(single_frame_pred <= 1)
        assert torch.all(all_frames_pred >= 0) and torch.all(all_frames_pred <= 1)
        
        # Check shapes
        assert single_frame_pred.shape == (1, 100, 1)
        assert all_frames_pred.shape == (1, 100, 1)
    
    def test_predict_frames_with_real_video(self, model):
        """Test predict_frames with real video data"""
        if not os.path.exists(TEST_VIDEO_PATH):
            pytest.skip("Test video not available")
        
        # Load video frames using the model's predict_video method
        with torch.no_grad():
            video_frames, single_frame_pred, all_frames_pred = model.predict_video(TEST_VIDEO_PATH, quiet=True)
        
        # Check that we got some frames
        assert len(video_frames) > 0
        assert len(single_frame_pred) == len(video_frames)
        assert len(all_frames_pred) == len(video_frames)
        
        # Check that predictions are probabilities
        assert torch.all(single_frame_pred >= 0) and torch.all(single_frame_pred <= 1)
        assert torch.all(all_frames_pred >= 0) and torch.all(all_frames_pred <= 1)
    
    def test_predictions_to_scenes(self):
        """Test predictions_to_scenes static method"""
        # Create dummy predictions with clear scene boundaries
        predictions = np.array([0.1, 0.1, 0.8, 0.1, 0.1, 0.9, 0.1, 0.1])
        
        scenes = TransNetV2.predictions_to_scenes(predictions, threshold=0.5)
        
        # Should detect scene boundaries at positions 2 and 5
        assert len(scenes) > 0
        assert scenes.dtype == np.int32
        
        # Check that scenes are valid (start <= end)
        for scene in scenes:
            assert scene[0] <= scene[1]
    
    def test_predictions_to_scenes_all_low(self):
        """Test predictions_to_scenes with all low predictions"""
        predictions = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        scenes = TransNetV2.predictions_to_scenes(predictions, threshold=0.5)
        
        # Should return one scene covering the whole video
        assert len(scenes) == 1
        assert scenes[0][0] == 0
        assert scenes[0][1] == len(predictions) - 1
    
    def test_predictions_to_scenes_all_high(self):
        """Test predictions_to_scenes with all high predictions"""
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        
        scenes = TransNetV2.predictions_to_scenes(predictions, threshold=0.5)
        
        # Should return one scene covering the whole video
        assert len(scenes) == 1
        assert scenes[0][0] == 0
        assert scenes[0][1] == len(predictions) - 1
    
    def test_model_with_different_devices(self):
        """Test model creation with different devices"""
        # Test CPU
        cpu_model = TransNetV2(device='cpu')
        assert cpu_model.device == torch.device('cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_model = TransNetV2(device='cuda')
            assert cuda_model.device == torch.device('cuda')
        
        # Test MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_model = TransNetV2(device='mps')
            assert mps_model.device == torch.device('mps')

if __name__ == '__main__':
    pytest.main([__file__]) 