"""Video processing and subtitle segment handling."""

import time
from types import SimpleNamespace
from typing import List
from utils import clean_subtitle_text


# Configuration constants
MIN_SEG_DURATION = 0.3
MAX_SEG_DURATION = 5.0   # Reduced from 7.0 for better readability
MAX_SEG_CHARS = 60       # Reduced from 80 for better screen fit
SEGMENT_BATCH_SIZE = 32


def split_long_segment(seg: SimpleNamespace) -> List[SimpleNamespace]:
    """Split overly long segments at natural break points."""
    text = seg.text.strip()
    duration = seg.end - seg.start
    
    # Check if segment needs splitting
    if duration <= MAX_SEG_DURATION and len(text) <= MAX_SEG_CHARS:
        return [seg]
    
    # Try multiple splitting strategies in order of preference
    split_points = []
    
    # Strategy 1: Strong punctuation (sentence endings)
    for punct in (".", "?", "!"):
        split_points.extend(_find_split_points(text, punct))
    
    # Strategy 2: Weak punctuation (clause boundaries)  
    for punct in (",", ";", ":"):
        split_points.extend(_find_split_points(text, punct))
    
    # Strategy 3: Conjunctions and connecting words
    for word in (" and ", " but ", " or ", " so ", " because ", " while ", " when ", " if "):
        split_points.extend(_find_split_points(text, word, word_boundary=True))
    
    # Strategy 4: Natural pauses (prepositions, articles)
    for word in (" that ", " which ", " who ", " where ", " with ", " from ", " to "):
        split_points.extend(_find_split_points(text, word, word_boundary=True))
    
    # Find the best split point closest to the middle
    if split_points:
        midpoint = len(text) // 2
        best_split = min(split_points, key=lambda x: abs(x - midpoint))
        
        # Only split if it creates reasonable-sized segments
        if 10 <= best_split <= len(text) - 10:
            left_text = text[:best_split].strip()
            right_text = text[best_split:].strip()
            
            # Calculate split timing based on character position
            split_ratio = len(left_text) / len(text) if len(text) > 0 else 0.5
            split_time = seg.start + (seg.end - seg.start) * split_ratio

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
            
            # Recursively split if segments are still too long
            result = []
            for subseg in [left_seg, right_seg]:
                result.extend(split_long_segment(subseg))
            return result
    
    # If no good split point found, return original segment
    return [seg]


def _find_split_points(text: str, delimiter: str, word_boundary: bool = False) -> List[int]:
    """Find all possible split points for a given delimiter."""
    split_points = []
    start = 0
    
    while True:
        idx = text.find(delimiter, start)
        if idx == -1:
            break
            
        if word_boundary:
            # For word boundaries, split after the word
            split_points.append(idx + len(delimiter))
        else:
            # For punctuation, split after the punctuation
            split_points.append(idx + 1)
            
        start = idx + 1
    
    return split_points


def merge_short_segments(segments: List[SimpleNamespace]) -> List[SimpleNamespace]:
    """Merge segments that are too short."""
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


def post_process_segments_batched(raw_segments: List[SimpleNamespace]) -> List[SimpleNamespace]:
    """Batch-process segments for better performance with text cleaning."""
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


def write_srt(segments: List[SimpleNamespace], out_path: str, use_word_ts: bool):
    """Write segments to SRT subtitle file."""
    from utils import format_timestamp
    
    with open(out_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if use_word_ts and hasattr(seg, "words") and seg.words:
                start_time = seg.words[0].start
                end_time = seg.words[-1].end
            else:
                start_time = seg.start
                end_time = seg.end

            start_ts = format_timestamp(start_time)
            end_ts = format_timestamp(end_time)
            f.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")
            idx += 1