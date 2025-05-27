import pytest
import os

def pytest_configure(config):
    """Configure pytest"""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(__file__))
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

@pytest.fixture(scope="session")
def test_video_path():
    """Provide path to test video"""
    return os.path.join(os.path.dirname(__file__), 'test.mp4') 