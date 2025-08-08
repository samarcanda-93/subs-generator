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

1. Place your video files (`.mp4`) in the project directory
2. Run the transcription script:

```bash
python transcribe.py
```

## Output

The script generates subtitle files in the `./out/` directory:

- `{filename}_orig.srt` - Original language subtitles
- `{filename}_en.srt` - English translation (if original is non-English)

## Configuration

Edit the configuration section in `transcribe.py`:

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