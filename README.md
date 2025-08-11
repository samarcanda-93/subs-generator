# Subtitle Generator

A Python/PyTorch suite that generates subtitles for video files. If the original audio is not in English, it also generates an English translation. Uses faster-whisper (OpenAI Whisper) for transcription and translation.

## Features

- Automatic speech recognition with Whisper large-v3 model
- Multi-language support with language auto-detection
- Automatic English translation for non-English content
- GPU acceleration support for faster processing
- Intelligent subtitle segmentation (splits long segments, merges short ones)
- Word-level timestamp accuracy
- SRT subtitle format output

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Base Requirements

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch (GPU-specific)

Choose the appropriate installation based on your GPU:

#### For RTX 40 Series and Older GPUs (RTX 30, 20, 10, GTX series)

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### For RTX 50 Series GPUs (RTX 5090, 5080, 5070 Ti, 5070)

RTX 50 series GPUs have CUDA compute capability sm_120 which requires special setup:

1. **Install PyTorch Nightly:**
   ```bash
   pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu124
   ```

2. **Download cuDNN 9.x:**
   - Visit [NVIDIA Developer cuDNN page](https://developer.nvidia.com/cudnn)
   - Download cuDNN 9.x for CUDA 12.x
   - Extract the following DLL files to your project root directory:
     - `cudnn64_9.dll`
     - `cudnn_adv64_9.dll`
     - `cudnn_cnn64_9.dll`
     - `cudnn_engines_precompiled64_9.dll`
     - `cudnn_engines_runtime_compiled64_9.dll`
     - `cudnn_graph64_9.dll`
     - `cudnn_heuristic64_9.dll`
     - `cudnn_ops64_9.dll`

#### For CPU-only Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Basic Usage

**Process specific video files:**
```bash
python transcribe.py video1.mp4 video2.mp4
```

**Process all videos in current directory:**
```bash
python transcribe.py
```

**Process videos from another directory:**
```bash
python transcribe.py --dir ./my_videos
```

### Advanced Usage

**Concurrent Processing (Multiple Files Simultaneously):**
```bash
# Process 4 files at once (faster, requires more VRAM)
python transcribe.py --concurrent 4 *.mp4

# Use smaller model for more concurrent files
python transcribe.py --model medium --concurrent 8 *.mp4
```

**Other Options:**
```bash
# Custom output directory
python transcribe.py --output ./subtitles video.mp4

# CPU-only processing (single file at a time)
python transcribe.py --cpu-only video.mp4

# Skip English translation (original language only)
python transcribe.py --skip-translation video.mp4

# English-only subtitles (translate everything to English, skip original)
python transcribe.py --english-only video.mp4

# Use different Whisper model size
python transcribe.py --model medium video.mp4

# View all available options
python transcribe.py --help
```

### Complete CLI Options

For a complete list of options, run:
```bash
python transcribe.py --help
```

**Available Options:**
- `--concurrent, -c`: Number of files to process simultaneously (default: 2)
- `--model, -m`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`)
- `--output, -o`: Output directory for subtitle files (default: `./out`)
- `--dir, -d`: Directory to scan for video files
- `--batch-size, -b`: Segment batch size for processing (default: 32)
- `--cpu-only`: Force CPU-only processing
- `--skip-translation`: Skip English translation for non-English audio
- `--english-only`: Generate only English subtitles (translate all audio)

## Output

The script generates subtitle files in the `./out/` directory:

- `{filename}_orig.srt` - Original language subtitles
- `{filename}_en.srt` - English translation (if original is non-English)

## Performance & Optimization

### Concurrent Processing Recommendations

**Based on GPU VRAM and Model Size:**

| GPU VRAM | Model | Max Concurrent Files | Command Example |
|----------|-------|---------------------|-----------------|
| 8GB (RTX 3070, 4060 Ti) | large-v3 | 1-2 files | `--concurrent 2` |
| 12GB (RTX 3080 Ti, 4070 Ti) | large-v3 | 2-3 files | `--concurrent 3` |
| 16GB (RTX 4080, 5070 Ti) | large-v3 | 3-4 files | `--concurrent 4` |
| 24GB (RTX 4090, 5090) | large-v3 | 5-6 files | `--concurrent 6` |
| Any GPU | medium | 2x more than large-v3 | `--model medium --concurrent 6` |
| Any GPU | small/tiny | 4x more than large-v3 | `--model small --concurrent 12` |

**CPU-Only Processing:**
- **Limitation**: CPU mode processes only **1 file at a time** (no concurrent processing)
- **Performance**: 5-10x slower than GPU processing
- **Usage**: `python transcribe.py --cpu-only video.mp4`

### Performance Tips

1. **Start conservatively** with default `--concurrent 2`
2. **Monitor GPU memory** usage and increase gradually
3. **Use smaller models** (medium/small) for more concurrent processing
4. **Out of memory?** Reduce concurrent files or use smaller model

## Configuration

You can configure processing through command-line arguments (recommended) or edit `transcribe.py`:

**Command-line configuration:**
```bash
python transcribe.py --model medium --concurrent 4 --output ./subs *.mp4
```

**File-based configuration** in `transcribe.py`:
```python
MODEL_NAME       = "large-v3"        # Whisper model size
COMPUTE_TYPE     = "float16"         # GPU: float16, CPU: int8
INPUT_DIR        = "."               # Input video directory
OUTPUT_DIR       = "./out"           # Output subtitle directory
MIN_SEG_DURATION = 0.3               # Minimum segment duration (seconds)
MAX_SEG_DURATION = 7.0               # Maximum segment duration (seconds)
MAX_SEG_CHARS    = 80                # Maximum characters per segment
```

## GPU Support Notes

- **RTX 50 series**: May show compatibility warnings but will work correctly with cuDNN files in place
- **RTX 40 and older**: Full native support with stable PyTorch versions
- **Multiple GPUs**: Uses first available GPU by default

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- FFmpeg (for audio extraction)
- ~4GB VRAM for large-v3 model on GPU

## Troubleshooting

### RTX 50 Series "sm_120 not compatible" Warning

This warning is expected and can be ignored. The script will work correctly with the cuDNN files in your project directory.

### Out of Memory Errors

Try reducing the model size or using CPU processing:
- Change `MODEL_NAME` to `"large"`, `"medium"`, or `"small"`
- Use CPU: set `device="cpu"` and `compute_type="int8"`

### Missing cuDNN Files

For RTX 50 series users, ensure all cuDNN 9.x DLL files are in your project root directory alongside `transcribe.py`.