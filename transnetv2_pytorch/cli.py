import time
import os
# Set MPS fallback IMMEDIATELY, before any PyTorch imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
import pandas as pd
import argparse
import json
import sys
import warnings
from tqdm import tqdm
import importlib.resources as resources

# Silence MPS fallback warnings for CLI usage
warnings.filterwarnings("ignore", message=".*MPS backend.*will fall back to run on the CPU.*")

def detect_best_device():
    """
    Automatically detect the best available device
    Priority: CUDA > MPS > CPU
    """
    # Import here to use the enhanced TransNetV2 method
    from .transnetv2_pytorch import TransNetV2
    device_str = TransNetV2._detect_best_device()
    return torch.device(device_str)

def get_device(device_arg):
    """
    Get the appropriate device based on user input
    """
    if device_arg == 'auto':
        device = detect_best_device()
        return device
    else:
        device = torch.device(device_arg)
        return device

def get_video_fps(video_path, verbose=False):
    """
    Extract FPS from video file using ffmpeg
    DEPRECATED: Use TransNetV2.get_video_fps() instead
    """
    # Import here to use the enhanced TransNetV2 method
    from .transnetv2_pytorch import TransNetV2
    model = TransNetV2()
    return model.get_video_fps(video_path)

def frame_to_timestamp(frame_number, fps):
    """
    Convert frame number to timestamp in ss.mmm format
    DEPRECATED: Use TransNetV2.frame_to_timestamp() instead
    """
    # Import here to use the enhanced TransNetV2 method
    from .transnetv2_pytorch import TransNetV2
    return TransNetV2.frame_to_timestamp(frame_number, fps)

def save_results(data, output_path, format_type):
    """
    Save results in the specified format
    """
    if format_type == 'csv':
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    elif format_type == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif format_type == 'jsonl':
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def process_video_to_output(model, video_path, output_path, format_type, threshold=0.5, verbose=False, quiet=False, no_progress_bar=False):
    """
    Process video and save scene transitions in the specified format
    """
    if not quiet:
        print(f"Processing video: {video_path}")
    
    start_time = time.time()
    
    # Create progress bar for overall process (verbose only and not suppressed)
    pbar = None if (quiet or not verbose or no_progress_bar) else tqdm(total=2, desc="Processing", unit="step")
    
    if pbar:
        pbar.set_description("Running inference")
    
    try:
        # Use the enhanced method that provides all the rich data with the specified threshold
        results = model.analyze_video(
            video_path, 
            threshold=threshold,
            quiet=(quiet or no_progress_bar)
        )
    except RuntimeError as e:
        if "MPS backend out of memory" in str(e):
            print("Error: MPS backend out of memory.")
            print("The model has applied automatic memory optimizations, but your video may still exceed available memory.")
            print("Try reducing video resolution or splitting longer videos into segments.")
            raise RuntimeError("TransNetV2 prediction failed: MPS backend out of memory.") from e
        else:
            # Re-raise other runtime errors
            raise
    
    if pbar:
        pbar.set_description("Generating output")
        pbar.update(1)
    
    inference_time = time.time() - start_time
    if verbose:
        tqdm.write(f"Inference time: {inference_time:.2f} seconds")
        tqdm.write(f"Found {results['total_scenes']} scenes")
        tqdm.write(f"Video FPS: {results['fps']:.3f}")
    
    # The scenes data is already in the perfect format for output
    output_data = results['scenes']
    
    # Save results in the specified format
    save_results(output_data, output_path, format_type)
    
    if pbar:
        pbar.update(1)
        pbar.close()
    
    if not quiet:
        print(f"Results saved to: {output_path} (format: {format_type})")
        print(f"Output contains {len(output_data)} scene transitions")
    
    return output_data

def get_default_weights_path():
    """
    Get the path to the default weights file in the package
    """
    try:
        # Try to get the weights file from the package using importlib.resources
        with resources.path('transnetv2_pytorch', 'transnetv2-pytorch-weights.pth') as weights_path:
            return str(weights_path)
    except:
        # Fallback to looking in the package directory
        package_dir = os.path.dirname(__file__)
        weights_path = os.path.join(package_dir, 'transnetv2-pytorch-weights.pth')
        if os.path.exists(weights_path):
            return weights_path
        
        # Final fallback to current directory
        return 'transnetv2-pytorch-weights.pth'

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video for scene detection using TransNetV2')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for inference (default: auto)')
    parser.add_argument('video', nargs='?', default=None,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output file (default: auto-generated based on input)')
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'json', 'jsonl'],
                       help='Output format (default: csv)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights file (default: use bundled weights)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Scene boundary detection threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--no-progress-bar', action='store_true',
                       help='Suppress progress bar (useful for scripted applications)')
    
    # Mutually exclusive verbose/quiet options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument('--verbose', '-v', action='store_true',
                                help='Enable verbose output with detailed messages')
    verbosity_group.add_argument('--quiet', '-q', action='store_true',
                                help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Check if format was explicitly specified by user
    format_explicitly_specified = '--format' in sys.argv
    
    # Check if video file is provided
    if args.video is None:
        parser.error("Video file is required. Usage: python -m transnetv2_pytorch <video_file>")
    
    # Set default weights path if not specified
    if args.weights is None:
        args.weights = get_default_weights_path()
    
    # Auto-generate output filename if not specified
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{video_name}_scenes.{args.format}"
    
    # Auto-detect format from output file extension if format not explicitly specified
    elif not format_explicitly_specified:
        ext = os.path.splitext(args.output)[1].lower()
        if ext == '.json':
            args.format = 'json'
        elif ext == '.jsonl':
            args.format = 'jsonl'
    
    # Setup device FIRST, before importing model
    device = get_device(args.device)
    if not args.quiet:
        print(f"Using device: {device}")
    
    # NOW import the model after setting environment
    from .transnetv2_pytorch import TransNetV2
    
    # Print device capabilities for debugging (verbose only)
    if args.verbose:
        tqdm.write(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            tqdm.write(f"MPS available: {torch.backends.mps.is_available()}")
        else:
            tqdm.write("MPS not supported in this PyTorch version")
        tqdm.write(f"Using weights: {args.weights}")
    
    # Initialize model with enhanced device handling
    if args.verbose:
        tqdm.write("Initializing model...")
    
    # Use the enhanced constructor that handles device auto-detection and memory optimization automatically
    model = TransNetV2(device=args.device if args.device != 'auto' else 'auto')
    
    # Load weights
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        if args.weights == get_default_weights_path():
            print("The bundled weights file is missing. Please ensure the package is properly installed.")
        else:
            print("Please check the path to your custom weights file.")
        sys.exit(1)
    
    if args.verbose:
        tqdm.write("Loading model weights...")
    
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Check video file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        sys.exit(1)
    
    with torch.no_grad():
        data = process_video_to_output(model, args.video, args.output, 
                                     args.format, args.threshold, args.verbose, args.quiet, args.no_progress_bar)
        
        # Print summary (verbose only)
        if args.verbose and not args.quiet:
            print(f"\nSummary:")
            print(f"Total scenes: {len(data)}")
            if len(data) > 0:
                probabilities = [item['probability'] for item in data]
                print(f"Probability range: {min(probabilities):.4f} - {max(probabilities):.4f}")
                
                # Show first few entries
                print(f"\nFirst 5 scenes:")
                for i, item in enumerate(data[:5]):
                    print(f"  Shot {item['shot_id']}: frames {item['start_frame']}-{item['end_frame']} "
                          f"({item['start_time']}s-{item['end_time']}s) prob={item['probability']:.4f}")

if __name__ == "__main__":
    main() 