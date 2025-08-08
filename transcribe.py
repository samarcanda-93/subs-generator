import os
import time
import torch
from faster_whisper import WhisperModel
from types import SimpleNamespace

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


def main():
    if not torch.cuda.is_available():
        print("Error: No GPU available. This code requires CUDA.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mp4_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    if not mp4_files:
        print("No .mp4 files found.")
        return

    for filename in mp4_files:
        base = os.path.splitext(filename)[0]
        orig_srt = os.path.join(OUTPUT_DIR, f"{base}_orig.srt")
        en_srt   = os.path.join(OUTPUT_DIR, f"{base}_en.srt")

        # Skip if original-language SRT already exists
        if os.path.exists(orig_srt):
            print(f"Skipping {filename}: '{orig_srt}' already exists.")
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        print(f"\nProcessing {filename}...")

        # 1) Transcribe original with word timestamps (default settings, no filters)
        print("  -> Transcribing original (default settings, word timestamps)...")
        t0 = time.time()
        segments_orig, info = model.transcribe(
            filepath,
            word_timestamps=True
        )
        t1 = time.time()
        detected_lang = info.language
        print(f"  * Original transcription done in {t1 - t0:.1f}s")
        print(f"  -> Detected language: {detected_lang}")

        # Post-process segments
        print("  -> Post-processing segments (splitting long, merging short)...")
        processed_orig = post_process_segments(segments_orig)

        # Write the original-language SRT
        print(f"  -> Writing original-language SRT to '{orig_srt}'")
        write_srt(processed_orig, orig_srt, use_word_ts=True)

        # 2) If non-English, translate with default settings
        if detected_lang != "en":
            print("  -> Detected non-English audio; starting translation to English...")
            t2 = time.time()
            segments_trans, _ = model.transcribe(
                filepath,
                task="translate",
                word_timestamps=True
            )
            t3 = time.time()
            print(f"  * Translation done in {t3 - t2:.1f}s")

            processed_trans = post_process_segments(segments_trans)
            print(f"  -> Writing English translation SRT to '{en_srt}'")
            write_srt(processed_trans, en_srt, use_word_ts=True)
        else:
            print("  -> Audio is already English; skipping translation.")

        print(f"Finished {filename}.")


if __name__ == "__main__":
    main()
