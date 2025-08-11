"""Command-line interface and argument parsing."""

import argparse
from pathlib import Path
from typing import List


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles for video files using Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py video1.mp4 video2.mp4        # Process specific files
  python transcribe.py --dir ./videos               # Process all videos in directory
  python transcribe.py *.mp4                        # Process all .mp4 in current dir
  python transcribe.py --concurrent 4 video.mp4     # Use 4 concurrent workers
        """
    )
    
    parser.add_argument(
        'videos', 
        nargs='*', 
        help='Video files to process (supports .mp4, .avi, .mkv, .mov, .flv)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory to scan for video files (default: current directory if no videos specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./out',
        help='Output directory for subtitle files (default: ./out)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper model size (default: large-v3)'
    )
    
    parser.add_argument(
        '--concurrent', '-c',
        type=int,
        default=2,
        help='Maximum concurrent file processing (default: 2)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Segment batch size for processing (default: 32)'
    )
    
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only processing (slower but works without GPU)'
    )
    
    parser.add_argument(
        '--skip-translation',
        action='store_true',
        help='Skip English translation even for non-English audio'
    )
    
    parser.add_argument(
        '--english-only',
        action='store_true',
        help='Generate only English subtitles (translate all audio to English, skip original language)'
    )
    
    parser.add_argument(
        '--translate-to',
        type=str,
        default='en',
        help='Target language for translation (default: en). Examples: it (Italian), fr (French), de (German), es (Spanish), ru (Russian), zh (Chinese), ja (Japanese), ko (Korean)'
    )
    
    parser.add_argument(
        '--quality-check',
        action='store_true',
        help='Enable AI-powered translation quality assessment and correction suggestions (requires GEMINI_API_KEY or HUGGINGFACE_API_KEY)'
    )
    
    parser.add_argument(
        '--auto-correct',
        action='store_true',
        help='Automatically offer corrections for low-quality translations (implies --quality-check)'
    )
    
    parser.add_argument(
        '--overwrite-corrections',
        action='store_true',
        help='Overwrite original files when applying corrections (default: create new *_ai_corrected.srt files)'
    )
    
    return parser.parse_args()


def get_video_files(args) -> List[str]:
    """Get list of video files to process based on arguments."""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.m4v'}
    video_files = []
    
    # If specific videos are provided as arguments
    if args.videos:
        for video_arg in args.videos:
            video_path = Path(video_arg)
            if video_path.exists() and video_path.is_file():
                if video_path.suffix.lower() in video_extensions:
                    video_files.append(str(video_path))
                else:
                    print(f"Warning: '{video_arg}' is not a supported video format")
            else:
                print(f"Warning: '{video_arg}' does not exist or is not a file")
    
    # If directory is specified or no videos provided, scan directory
    elif args.dir or not args.videos:
        scan_dir = Path(args.dir) if args.dir else Path('.')
        
        if not scan_dir.exists():
            print(f"Error: Directory '{scan_dir}' does not exist")
            return []
        
        for ext in video_extensions:
            video_files.extend([str(p) for p in scan_dir.glob(f"*{ext}")])
    
    return sorted(video_files)