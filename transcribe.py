import os
import time
import asyncio
import concurrent.futures
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from faster_whisper import WhisperModel
from types import SimpleNamespace
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import threading

# Add current directory to PATH for cuDNN files
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = current_dir + os.pathsep + os.environ.get('PATH', '')

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME       = "large-v3"
COMPUTE_TYPE     = "float16"       # float16 on GPU
INPUT_DIR        = "."             # where your .mp4 files live
OUTPUT_DIR       = "./out"         # where to write .srt files

# Thresholds for splitting/merging
MIN_SEG_DURATION = 0.3
MAX_SEG_DURATION = 7.0
MAX_SEG_CHARS    = 80

# Async processing configuration
MAX_CONCURRENT_FILES = 2           # Number of files to process simultaneously
SEGMENT_BATCH_SIZE   = 32          # Batch size for segment processing


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def split_long_segment(seg: SimpleNamespace) -> list[SimpleNamespace]:
    text = seg.text.strip()
    duration = seg.end - seg.start
    if duration <= MAX_SEG_DURATION and len(text) <= MAX_SEG_CHARS:
        return [seg]

    midpoint = len(text) // 2
    for punct in (".", "?", "!"):
        idx = text.rfind(punct, 0, midpoint)
        if idx == -1:
            idx = text.find(punct, midpoint)
        if idx != -1:
            left_text  = (text[: idx + 1]).strip()
            right_text = (text[idx + 1 :]).strip()
            split_ratio = (len(left_text) / len(text)) if len(text) > 0 else 0.5
            split_time  = seg.start + (seg.end - seg.start) * split_ratio

            left_seg = SimpleNamespace(
                start=seg.start,
                end=split_time,
                text=left_text,
                words=None
            )
            right_seg = SimpleNamespace(
                start=split_time,
                end=seg.end,
                text=right_text,
                words=None
            )
            return [left_seg, right_seg]
    return [seg]


def merge_short_segments(segments: list[SimpleNamespace]) -> list[SimpleNamespace]:
    merged = []
    skip_next = False
    for i, seg in enumerate(segments):
        if skip_next:
            skip_next = False
            continue
        duration = seg.end - seg.start
        if duration < MIN_SEG_DURATION and i + 1 < len(segments):
            next_seg = segments[i + 1]
            combined_duration = next_seg.end - seg.start
            if combined_duration <= MAX_SEG_DURATION:
                new_text = (seg.text + " " + next_seg.text).strip()
                merged_seg = SimpleNamespace(
                    start=seg.start,
                    end=next_seg.end,
                    text=new_text,
                    words=None
                )
                merged.append(merged_seg)
                skip_next = True
                continue
        merged.append(seg)
    return merged


def post_process_segments(raw_segments: list[SimpleNamespace]) -> list[SimpleNamespace]:
    split_list: list[SimpleNamespace] = []
    for seg in raw_segments:
        split_list.extend(split_long_segment(seg))
    processed = merge_short_segments(split_list)
    return processed


def write_srt(segments: list[SimpleNamespace], out_path: str, use_word_ts: bool):
    with open(out_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if use_word_ts and hasattr(seg, "words") and seg.words:
                start_time = seg.words[0].start
                end_time   = seg.words[-1].end
            else:
                start_time = seg.start
                end_time   = seg.end

            start_ts = format_timestamp(start_time)
            end_ts   = format_timestamp(end_time)
            f.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")
            idx += 1


async def process_file(video_path: str, filename: str, executor: concurrent.futures.ThreadPoolExecutor, 
                      device: str, compute_type: str, skip_translation: bool, english_only: bool = False) -> Tuple[str, bool, Optional[str]]:
    """Process a single video file asynchronously."""
    base = os.path.splitext(filename)[0]
    orig_srt = os.path.join(OUTPUT_DIR, f"{base}_orig.srt")

    # Skip if original-language SRT already exists
    if os.path.exists(orig_srt):
        return filename, True, f"Skipped: '{orig_srt}' already exists"
    
    try:
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        # Create progress callback for this file
        def progress_update(step, total_steps):
            pass  # We'll implement this with a better system
            
        result = await loop.run_in_executor(
            executor, _transcribe_file_sync, video_path, filename, device, compute_type, skip_translation, english_only, progress_update
        )
        return filename, True, result
    except Exception as e:
        return filename, False, f"Error: {str(e)}"


def _transcribe_file_sync(video_path: str, filename: str, device: str, compute_type: str, skip_translation: bool, english_only: bool = False, progress_callback=None) -> str:
    """Synchronous transcription function to run in thread pool."""
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    
    base = os.path.splitext(filename)[0]
    orig_srt = os.path.join(OUTPUT_DIR, f"{base}_orig.srt")
    en_srt   = os.path.join(OUTPUT_DIR, f"{base}_en.srt")
    
    print(f"\n[{filename}] Processing...")

    if english_only:
        # English-only mode: directly translate to English
        print(f"[{filename}] -> English-only mode: translating to English...")
        t0 = time.time()
        segments_eng, info = model.transcribe(
            video_path,
            task="translate",
            word_timestamps=True
        )
        t1 = time.time()
        detected_lang = info.language
        print(f"[{filename}] * English translation done in {t1 - t0:.1f}s")
        print(f"[{filename}] -> Detected language: {detected_lang}")

        # Post-process segments with batching
        print(f"[{filename}] -> Post-processing segments...")
        processed_eng = post_process_segments_batched(list(segments_eng))

        # Write the English SRT
        print(f"[{filename}] -> Writing English SRT")
        write_srt(processed_eng, en_srt, use_word_ts=True)
        
        return f"Completed successfully (English-only)"
    else:
        # Standard mode: original + optional translation
        # 1) Transcribe original with word timestamps
        print(f"[{filename}] -> Transcribing original...")
        t0 = time.time()
        segments_orig, info = model.transcribe(
            video_path,
            word_timestamps=True
        )
        t1 = time.time()
        detected_lang = info.language
        print(f"[{filename}] * Original transcription done in {t1 - t0:.1f}s")
        print(f"[{filename}] -> Detected language: {detected_lang}")

        # Post-process segments with batching
        print(f"[{filename}] -> Post-processing segments...")
        processed_orig = post_process_segments_batched(list(segments_orig))

        # Write the original-language SRT
        print(f"[{filename}] -> Writing original SRT")
        write_srt(processed_orig, orig_srt, use_word_ts=True)

    # 2) If non-English, translate
    if detected_lang != "en" and not skip_translation:
        print(f"[{filename}] -> Translating to English...")
        t2 = time.time()
        segments_trans, _ = model.transcribe(
            video_path,
            task="translate",
            word_timestamps=True
        )
        t3 = time.time()
        print(f"[{filename}] * Translation done in {t3 - t2:.1f}s")

        processed_trans = post_process_segments_batched(list(segments_trans))
        print(f"[{filename}] -> Writing English translation SRT")
        write_srt(processed_trans, en_srt, use_word_ts=True)
    else:
        print(f"[{filename}] -> Audio is English; skipping translation")

    return f"Completed successfully"


def clean_subtitle_text(text: str) -> Optional[str]:
    """Remove subtitle artifacts, credits, and watermarks."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    text_lower = text.lower()
    
    # Common subtitle artifacts to filter out
    artifacts = [
        "translated by",
        "subtitles by",
        "subtitle by",
        "captioned by",
        "transcribed by",
        "www.",
        "http",
        "subscribe",
        "like and subscribe",
        "follow us",
        "© ",
        "copyright",
        "all rights reserved",
        "opensubtitles",
        "addic7ed",
        "subscene",
        "yify",
        "rarbg"
    ]
    
    # Filter out lines containing artifacts
    for artifact in artifacts:
        if artifact in text_lower:
            return None
    
    # Filter out very short non-meaningful segments
    if len(text) < 3:
        return None
    
    # Filter out lines that are mostly symbols/numbers
    if len([c for c in text if c.isalpha()]) / len(text) < 0.5:
        return None
    
    return text


def post_process_segments_batched(raw_segments: List[SimpleNamespace]) -> List[SimpleNamespace]:
    """Batch-process segments for better performance with text cleaning."""
    # Process segments in batches
    all_split_segments = []
    
    for i in range(0, len(raw_segments), SEGMENT_BATCH_SIZE):
        batch = raw_segments[i:i + SEGMENT_BATCH_SIZE]
        batch_split = []
        
        for seg in batch:
            # Clean the text first
            cleaned_text = clean_subtitle_text(seg.text)
            if cleaned_text:
                # Create new segment with cleaned text
                cleaned_seg = SimpleNamespace(
                    start=seg.start,
                    end=seg.end,
                    text=cleaned_text,
                    words=seg.words if hasattr(seg, 'words') else None
                )
                batch_split.extend(split_long_segment(cleaned_seg))
        
        all_split_segments.extend(batch_split)
        
        # Optional: Add small delay between batches to prevent overwhelming GPU
        if i + SEGMENT_BATCH_SIZE < len(raw_segments):
            time.sleep(0.01)  # 10ms delay
    
    # Merge short segments (this is sequential as it depends on adjacent segments)
    return merge_short_segments(all_split_segments)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles for video files using Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py video1.mp4 video2.mp4        # Process specific files
  python transcribe.py --dir ./videos               # Process all .mp4 in directory
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


async def main():
    args = parse_arguments()
    
    # Update global configuration from arguments
    global MODEL_NAME, OUTPUT_DIR, MAX_CONCURRENT_FILES, SEGMENT_BATCH_SIZE
    MODEL_NAME = args.model
    OUTPUT_DIR = args.output
    MAX_CONCURRENT_FILES = args.concurrent
    SEGMENT_BATCH_SIZE = args.batch_size
    
    # Check GPU availability
    if not args.cpu_only and not torch.cuda.is_available():
        print("Warning: No GPU available. Falling back to CPU processing.")
        args.cpu_only = True
    
    if args.cpu_only:
        print("Using CPU processing (slower but more compatible)")
        print("Note: CPU mode processes only 1 file at a time (concurrent processing disabled)")
        device = "cpu"
        compute_type = "int8"
        # Force single file processing for CPU mode
        MAX_CONCURRENT_FILES = 1
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        device = "cuda"
        compute_type = "float16"
    
    print(f"Model: {MODEL_NAME}")
    print(f"Async processing: max {MAX_CONCURRENT_FILES} concurrent files")
    print(f"Segment batch size: {SEGMENT_BATCH_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get video files to process
    video_files = get_video_files(args)
    
    if not video_files:
        print("No video files found to process.")
        print("Use --help for usage examples.")
        return

    print(f"\nFound {len(video_files)} video file(s) to process:")
    for video in video_files:
        print(f"  - {video}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create thread pool executor for CPU-bound transcription tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as executor:
        # Create semaphore to limit concurrent GPU operations
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
        
        async def process_with_semaphore(video_path):
            async with semaphore:
                filename = os.path.basename(video_path)
                return await process_file(video_path, filename, executor, device, compute_type, args.skip_translation, args.english_only)
        
        # Process all files concurrently with limited concurrency using tqdm
        print(f"\nStarting concurrent processing...")
        start_time = time.time()
        
        # Create a progress tracking system
        completed_files = {"count": 0}
        progress_bar = tqdm(total=len(video_files), desc="Processing videos", unit="file", 
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}")
        
        # Progress heartbeat updater
        def progress_heartbeat():
            dots = 0
            while completed_files["count"] < len(video_files):
                dots = (dots + 1) % 4
                activity = "●" * dots + "○" * (3 - dots)
                progress_bar.set_postfix_str(f"Active {activity}")
                time.sleep(0.5)
        
        # Start heartbeat in separate thread
        heartbeat_thread = threading.Thread(target=progress_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        async def track_progress(coro):
            result = await coro
            completed_files["count"] += 1
            progress_bar.update(1)
            progress_bar.set_postfix_str("Done" if completed_files["count"] == len(video_files) else "Processing...")
            return result
        
        tasks = [track_progress(process_with_semaphore(video_path)) for video_path in video_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop heartbeat and close progress bar
        completed_files["count"] = len(video_files)
        progress_bar.set_postfix_str("Complete")
        progress_bar.close()
        
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
