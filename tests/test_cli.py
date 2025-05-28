import pytest
import os
import tempfile
import json
import pandas as pd
from unittest.mock import patch
import subprocess
import sys
import torch

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

class TestCLI:
    """Test the CLI functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for tests"""
        self.original_env = os.environ.copy()
        os.environ['TRANSNET_TEST_DEVICE'] = get_test_device()
        yield
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_video_file_exists(self):
        """Test that the test video file exists"""
        assert os.path.exists(TEST_VIDEO_PATH), "Test video file is missing"
    
    def test_cli_help(self):
        """Test that CLI help works"""
        result = subprocess.run([
            sys.executable, '-m', 'transnetv2_pytorch', '--help'
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Process video for scene detection' in result.stdout
    
    def test_cli_no_video_error(self):
        """Test that CLI shows error when no video is provided"""
        result = subprocess.run([
            sys.executable, '-m', 'transnetv2_pytorch'
        ], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'Video file is required' in result.stderr
    
    def test_cli_csv_output(self):
        """Test CLI with CSV output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.csv')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--quiet'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_file), "Output file was not created"
            
            # Check CSV content
            df = pd.read_csv(output_file)
            assert 'shot_id' in df.columns
            assert 'start_frame' in df.columns
            assert 'end_frame' in df.columns
            assert 'start_time' in df.columns
            assert 'end_time' in df.columns
            assert 'probability' in df.columns
            assert len(df) > 0, "No scenes detected"
            
            # Check that shot_id starts from 1
            assert df['shot_id'].min() == 1
            
            # Check that frames are in order
            assert all(df['start_frame'] <= df['end_frame'])
    
    def test_cli_json_output(self):
        """Test CLI with JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.json')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--format', 'json',
                '--quiet'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_file), "Output file was not created"
            
            # Check JSON content
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) > 0, "No scenes detected"
            
            # Check first scene structure
            scene = data[0]
            required_fields = ['shot_id', 'start_frame', 'end_frame', 'start_time', 'end_time', 'probability']
            for field in required_fields:
                assert field in scene, f"Missing field: {field}"
            
            assert scene['shot_id'] == 1
    
    def test_cli_jsonl_output(self):
        """Test CLI with JSONL output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.jsonl')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--format', 'jsonl',
                '--quiet'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_file), "Output file was not created"
            
            # Check JSONL content
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) > 0, "No scenes detected"
            
            # Check first line
            first_scene = json.loads(lines[0])
            required_fields = ['shot_id', 'start_frame', 'end_frame', 'start_time', 'end_time', 'probability']
            for field in required_fields:
                assert field in first_scene, f"Missing field: {field}"
    
    def test_cli_auto_output_filename(self):
        """Test that CLI auto-generates output filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test video to temp dir with a specific name
            test_video = os.path.join(tmpdir, 'my_video.mp4')
            import shutil
            shutil.copy2(TEST_VIDEO_PATH, test_video)
            
            # Run CLI without specifying output
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                test_video,
                '--quiet'
            ], capture_output=True, text=True, cwd=tmpdir)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            
            # Check that auto-generated file exists
            expected_output = os.path.join(tmpdir, 'my_video_scenes.csv')
            assert os.path.exists(expected_output), "Auto-generated output file not found"
    
    def test_cli_format_auto_detection(self):
        """Test that CLI auto-detects format from file extension"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.json')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--quiet'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_file), "Output file was not created"
            
            # Should be JSON format even though we didn't specify --format json
            with open(output_file, 'r') as f:
                data = json.load(f)
            assert isinstance(data, list)
    
    def test_cli_verbose_mode(self):
        """Test CLI verbose mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.csv')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--verbose'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert 'Using device:' in result.stdout
            assert 'Summary:' in result.stdout
            assert 'Total scenes:' in result.stdout
    
    def test_cli_no_progress_bar(self):
        """Test CLI with --no-progress-bar flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.csv')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--no-progress-bar'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            # Should not contain progress bar characters
            assert '|' not in result.stdout  # Progress bar uses | characters
            assert '%' not in result.stdout  # Progress percentages
    
    def test_cli_device_selection(self):
        """Test CLI device selection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.csv')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--device', 'cpu',
                '--verbose'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert 'Using device: cpu' in result.stdout
    
    def test_cli_nonexistent_video(self):
        """Test CLI with non-existent video file"""
        result = subprocess.run([
            sys.executable, '-m', 'transnetv2_pytorch',
            'nonexistent_video.mp4',
            '--quiet'
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert 'Video file not found' in result.stdout or 'Video file not found' in result.stderr
    
    def test_cli_custom_weights_file(self):
        """Test CLI with custom weights file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.csv')
            fake_weights = os.path.join(tmpdir, 'fake_weights.pth')
            
            # Create a fake weights file
            with open(fake_weights, 'w') as f:
                f.write("fake weights")
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--weights', fake_weights,
                '--quiet'
            ], capture_output=True, text=True)
            
            # Should fail because weights file is invalid, but should try to load it
            assert result.returncode != 0
    
    def test_cli_nonexistent_weights_file(self):
        """Test CLI with non-existent weights file"""
        result = subprocess.run([
            sys.executable, '-m', 'transnetv2_pytorch',
            TEST_VIDEO_PATH,
            '--weights', 'nonexistent_weights.pth',
            '--quiet'
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert 'Weights file not found' in result.stdout or 'Weights file not found' in result.stderr
    
    def test_cli_auto_detect_jsonl_format(self):
        """Test CLI auto-detects JSONL format from file extension"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.jsonl')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--quiet'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_file), "Output file was not created"
            
            # Should be JSONL format even though we didn't specify --format jsonl
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) > 0
            # Each line should be valid JSON
            for line in lines:
                json.loads(line.strip())
    
    def test_cli_explicit_format_override(self):
        """Test that explicit format overrides file extension detection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_output.json')
            
            result = subprocess.run([
                sys.executable, '-m', 'transnetv2_pytorch',
                TEST_VIDEO_PATH,
                '--output', output_file,
                '--format', 'csv',  # Explicitly specify CSV despite .json extension
                '--quiet'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_file), "Output file was not created"
            
            # Should be CSV format despite .json extension
            df = pd.read_csv(output_file)
            assert 'shot_id' in df.columns

if __name__ == '__main__':
    pytest.main([__file__]) 