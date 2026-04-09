# Configuration Guide for SAM 2 Real-Time

This document details the configuration files, model hyperparameters, and environment setup required to run `segment-anything-2-real-time` efficiently across different hardware profiles.

## 1. Model Configuration YAMLs (`configs/sam2.1/`)

The architecture of the SAM 2 model is defined entirely via YAML configuration files located in the `configs/` directory. You must pair the correct YAML file with the corresponding downloaded `.pt` checkpoint.

### Available Profiles:

| Model Tier | YAML Config File | Checkpoint File | VRAM Req. | Target FPS (RTX 3090) | Use Case |
|------------|------------------|-----------------|-----------|-----------------------|----------|
| **Tiny** | `sam2.1_hiera_t.yaml` | `sam2.1_hiera_tiny.pt` | ~2 GB | 60+ FPS | Edge devices (Jetson), laptops. |
| **Small** | `sam2.1_hiera_s.yaml` | `sam2.1_hiera_small.pt`| ~3 GB | 45+ FPS | Standard webcams, balanced perf. |
| **Base+** | `sam2.1_hiera_b+.yaml`| `sam2.1_hiera_base_plus.pt`| ~5 GB | 30 FPS | High accuracy tracking. |
| **Large** | `sam2.1_hiera_l.yaml` | `sam2.1_hiera_large.pt`| ~8 GB | 15-20 FPS | Offline video processing, complex scenes. |

### Key YAML Hyperparameters:

Inside the YAML files, you will find settings defining the Memory Bank behavior. While generally you shouldn't alter the transformer block numbers, you can tweak the memory settings for real-time constraints:

- `max_cond_frames_in_attn`: (Default: `1`) The number of prompted (conditioned) frames to keep in memory. For real-time, keeping this at 1 is standard.
- `max_non_cond_frames_in_attn`: (Default: `7` or `15`) The size of the rolling temporal window. 
  - *Tuning*: If you are experiencing CUDA OOM errors during a long live stream, reduce this number (e.g., to `3`). This means the model will only "remember" the last 3 frames for temporal tracking. Tracking robustness might slightly decrease during heavy occlusions, but VRAM usage will drop significantly.

## 2. Environment Variables

- `CUDA_VISIBLE_DEVICES`: Standard PyTorch variable to restrict which GPU the model runs on.
  ```bash
  export CUDA_VISIBLE_DEVICES="0" # Force execution on GPU 0
  ```
- `PYTORCH_CUDA_ALLOC_CONF`: To prevent memory fragmentation during infinite loops, it is highly recommended to set this variable before running the demo scripts.
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

## 3. Command Line Flags

When executing the provided demo scripts (`demo/camera_tracking.py` or `demo/video_inference.py`), the following flags configure the runtime behavior:

### `camera_tracking.py`
- `--checkpoint` (Required): Path to the model weights.
- `--config` (Required): Path to the YAML config.
- `--camera_id` (Default: `0`): The integer ID of the video capture device. Change to `1` or `2` if you have multiple webcams connected.
- `--resolution` (Default: `640x480`): The resolution to request from the webcam. Higher resolutions will be internally resized by the SAM encoder but rendering them locally might cause UI lag.

### `video_inference.py`
- `--video` (Required): Path to the input video.
- `--bbox` (Required): Four space-separated integers `X_MIN Y_MIN X_MAX Y_MAX`.
- `--output` (Required): Path to the output video.
- `--fps` (Optional): Force the output video FPS. If omitted, it inherits the input video's FPS.

## 4. Hardware Optimization Recommendations

To maximize real-time performance:
1. **Always use bfloat16**: The codebase defaults to `torch.bfloat16`. Do not change this to `float32` unless your GPU explicitly does not support bfloat16 (e.g., GTX 10-series or older). If unsupported, use `float16`.
2. **PyTorch 2.0+**: Ensure you are using PyTorch 2.x to benefit from `torch.compile` and optimized flash attention kernels automatically utilized by SAM2's transformer blocks.
