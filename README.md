# TransNet V2: Shot Boundary Detection Neural Network (PyTorch)

This repository contains a PyTorch implementation of [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838).

This is a PyTorch reimplementation of the TransNetV2 model that produces identical results as the original TensorFlow version. The code is for inference only.

## Performance

Our reevaluation of other publicly available state-of-the-art shot boundary methods (F1 scores):

Model | ClipShots | BBC Planet Earth | RAI
--- | :---: | :---: | :---:
TransNet V2 | **77.9** | **96.2** | 93.9
[TransNet](https://arxiv.org/abs/1906.03363) [(github)](https://github.com/soCzech/TransNet) | 73.5 | 92.9 | **94.3**
[Hassanien et al.](https://arxiv.org/abs/1705.03281) [(github)](https://github.com/melgharib/DSBD) | 75.9 | 92.6 | 93.9
[Tang et al., ResNet baseline](https://arxiv.org/abs/1808.04234) [(github)](https://github.com/Tangshitao/ClipShots_basline) | 76.1 | 89.3 | 92.8

## Installation

```bash
pip install transnetv2-pytorch
```

Or install from source:
```bash
git clone https://github.com/allenday/transnetv2_pytorch.git
cd transnetv2_pytorch
pip install -e .
```

## Usage

### Command Line Interface

The package provides both a direct command and Python module execution:

```bash
# Direct command
transnetv2_pytorch path/to/video.mp4

# Python module execution
python -m transnetv2_pytorch path/to/video.mp4
```

#### CLI Arguments

```bash
# Basic usage
transnetv2_pytorch path/to/video.mp4

# Specify output file
transnetv2_pytorch path/to/video.mp4 --output predictions.txt

# Use specific device
transnetv2_pytorch path/to/video.mp4 --device cuda

# Get help for all options
transnetv2_pytorch --help
```

### Python API

```python
import torch
from transnetv2_pytorch import TransNetV2

# Initialize model
model = TransNetV2()
model.eval()

# Automatic device selection
if torch.cuda.is_available():
    model = model.cuda()
elif torch.backends.mps.is_available():
    model = model.to('mps')

with torch.no_grad():
    # Input shape: batch_size x video_frames x height x width x channels (RGB)
    input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
    
    # Move to same device as model
    input_video = input_video.to(next(model.parameters()).device)
    
    single_frame_pred, all_frame_pred = model(input_video)
    
    # Get predictions
    single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
    all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
    
    # Find shot boundaries (example)
    shot_boundaries = single_frame_pred > 0.5
```

#### Advanced Usage

```python
# Custom device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransNetV2(device=device)

# Load custom weights
model = TransNetV2()
state_dict = torch.load('custom_weights.pth', map_location='cpu')
model.load_state_dict(state_dict)

# Batch processing
batch_size = 4
for batch in video_batches:
    predictions = model(batch)
    # Process predictions...
```

## Device Support

This implementation supports:
- **CPU**: Works on all systems
- **CUDA**: For NVIDIA GPUs
- **MPS**: For Apple Silicon Macs (automatic fallback for unsupported operations)

The model automatically detects and uses the best available device. For MPS devices, unsupported operations (like 3D convolutions) automatically fall back to CPU.

## Original Work & Training

This PyTorch implementation is based on the original TensorFlow version. For:
- **Training code and datasets**
- **TensorFlow implementation** 
- **Weight conversion utilities**
- **Research replication**

Please visit the original repository: **[soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)**

## Credits

### Original Work

This PyTorch implementation is based on the original TensorFlow TransNet V2 by Tomáš Souček and Jakub Lokoč.

If found useful, please cite the original work:

```bibtex
@article{soucek2020transnetv2,
    title={TransNet V2: An effective deep network architecture for fast shot transition detection},
    author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Loko{\v{c}}, Jakub},
    year={2020},
    journal={arXiv preprint arXiv:2008.04838},
}
```

### PyTorch Implementation

This production-ready PyTorch package was developed by [Your Name] with significant improvements including:
- Complete PyTorch reimplementation for inference
- Cross-platform device support (CPU, CUDA, MPS)
- Command-line interface
- Package distribution and installation
- Comprehensive testing and error handling

### Related Papers

- ACM Multimedia paper of the older version: [A Framework for Effective Known-item Search in Video](https://dl.acm.org/doi/abs/10.1145/3343031.3351046)
- The older version paper: [TransNet: A deep network for fast detection of common shot transitions](https://arxiv.org/abs/1906.03363)

## License

MIT License

Original work Copyright (c) 2020 Tomáš Souček, Jakub Lokoč  
PyTorch implementation Copyright (c) 2025 Allen Day

See the original [TransNetV2 repository](https://github.com/soCzech/TransNetV2) for the original license.
