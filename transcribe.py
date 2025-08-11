#!/usr/bin/env python3
"""
Subs Generator - AI-Powered Video Subtitle Generation

A high-performance subtitle generation tool using OpenAI Whisper and PyTorch.
Supports concurrent processing, multiple languages, and automatic translation.
"""

import os
import sys
import time
import asyncio
import concurrent.futures
import warnings

# Suppress PyTorch CUDA compatibility warnings for RTX 50 series
warnings.filterwarnings("ignore", message=".*with CUDA capability sm_120 is not compatible.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import torch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from cli import parse_arguments, get_video_files
from core import process_file
from utils import setup_cudnn_path


async def main():
    """Main entry point for the subtitle generation tool."""
    # Set up cuDNN path for RTX 50 series compatibility
    setup_cudnn_path()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configuration
    model_name = args.model
    output_dir = args.output
    max_concurrent_files = args.concurrent
    
    # Check GPU availability
    if not args.cpu_only and not torch.cuda.is_available():
        print("Warning: No GPU available. Falling back to CPU processing.")
        args.cpu_only = True
    
    if args.cpu_only:
        print("Using CPU processing (slower but more compatible)")
        print("Note: CPU mode processes only 1 file at a time (concurrent processing disabled)")
        device = "cpu"
        compute_type = "int8"
        max_concurrent_files = 1  # Force single file processing for CPU mode
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        device = "cuda"
        compute_type = "float16"
    
    print(f"Model: {model_name}")
    print(f"Async processing: max {max_concurrent_files} concurrent files")
    print(f"Output directory: {output_dir}")
    
    # Get video files to process
    video_files = get_video_files(args)
    
    if not video_files:
        print("No video files found to process.")
        print("Use --help for usage examples.")
        return

    print(f"\nFound {len(video_files)} video file(s) to process:")
    for video in video_files:
        print(f"  - {video}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files concurrently
    print(f"\nStarting concurrent processing...")
    start_time = time.time()
    
    # Simple progress tracking without spinner
    print(f"Processing {len(video_files)} files...")
    
    # Create thread pool executor for CPU-bound transcription tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
        # Create semaphore to limit concurrent GPU operations
        semaphore = asyncio.Semaphore(max_concurrent_files)
        
        async def process_with_semaphore(video_path):
            async with semaphore:
                filename = os.path.basename(video_path)
                return await process_file(
                    video_path, filename, executor, device, compute_type,
                    args.skip_translation, args.english_only, args.translate_to, model_name, output_dir,
                    args.quality_check or args.auto_correct, args.auto_correct, args.overwrite_corrections
                )
        
        async def track_progress(coro):
            result = await coro
            return result
        
        # Process all files
        tasks = [track_progress(process_with_semaphore(video_path)) for video_path in video_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Processing completed
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print results summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Files processed: {len(video_files)}")
    
    successful = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[FAIL] {os.path.basename(video_files[i])}: {result}")
        else:
            filename, success, message = result
            if success:
                successful += 1
                print(f"[OK] {filename}: {message}")
            else:
                print(f"[FAIL] {filename}: {message}")
    
    print(f"\nSuccess rate: {successful}/{len(video_files)} files")
    if successful > 0:
        avg_time = total_time / successful
        print(f"Average time per file: {avg_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())