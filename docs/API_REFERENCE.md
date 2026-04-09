# Segment Anything 2 Real-Time API Reference

This document outlines the programmatic API for integrating SAM2 real-time tracking into your custom computer vision applications.

## 1. Core Builder API

### `sam2.build_sam.build_sam2_camera_predictor`

The primary factory function for creating a real-time tracking instance.

```python
def build_sam2_camera_predictor(
    config_file: str,
    ckpt_path: str,
    device: str = "cuda",
    apply_postprocessing: bool = True
) -> SAM2CameraPredictor
```

**Parameters:**
- `config_file` (str): Path to the YAML configuration file defining the model architecture (e.g., `configs/sam2.1/sam2.1_hiera_s.yaml`).
- `ckpt_path` (str): Path to the downloaded PyTorch checkpoint (`.pt` file).
- `device` (str, optional): The compute device. Default is `"cuda"`. Can be `"cpu"` or `"mps"`.
- `apply_postprocessing` (bool, optional): If True, applies hole-filling and island removal to the output masks. Default is `True`.

**Returns:**
- An instance of the custom `SAM2CameraPredictor` class.

## 2. `SAM2CameraPredictor` Class

This class manages the temporal state and executes the forward passes.

### Method: `init_state`

Initializes the memory bank with the first frame and the associated user prompts.

```python
def init_state(
    self,
    frame: np.ndarray,
    prompts: dict
) -> None
```

**Parameters:**
- `frame` (np.ndarray): The first video frame in RGB format, shape `(H, W, 3)`.
- `prompts` (dict): A dictionary mapping `instance_id` (int) to its prompt data.
  - Format:
    ```python
    {
        1: {
            "points": np.array([[x1, y1], [x2, y2]]), # Shape (N, 2)
            "labels": np.array([1, 0])                # 1 = positive, 0 = negative
        },
        2: {
            "bbox": np.array([x_min, y_min, x_max, y_max])
        }
    }
    ```

### Method: `track`

Processes a subsequent frame and returns the tracked masks.

```python
def track(self, frame: np.ndarray) -> dict
```

**Parameters:**
- `frame` (np.ndarray): The current video frame in RGB format, shape `(H, W, 3)`.

**Returns:**
- `dict`: A dictionary mapping `instance_id` to its predicted binary mask.
  - Format: `{ 1: mask_array_1, 2: mask_array_2 }`
  - `mask_array` is a boolean NumPy array of shape `(H, W)`.

### Method: `reset_state`

Clears the memory bank and resets the predictor to accept a new video stream.

```python
def reset_state(self) -> None
```

## 3. Demo Scripts (CLI Tools)

### `demo/camera_tracking.py`

Runs the interactive webcam tracking demo.

```bash
python demo/camera_tracking.py --checkpoint <path> --config <path> [--camera_id <int>]
```

**Arguments:**
- `--checkpoint`: Path to the SAM2 `.pt` weights.
- `--config`: Path to the corresponding YAML config.
- `--camera_id`: OpenCV device index (default: `0`).

### `demo/video_inference.py`

Processes an MP4 file offline using a bounding box prompt.

```bash
python demo/video_inference.py --video <path> --bbox <x1> <y1> <x2> <y2> --output <path>
```

**Arguments:**
- `--video`: Path to the input `.mp4` file.
- `--bbox`: Four integers defining the bounding box on the *first frame*.
- `--output`: Path to save the masked `.mp4` file.

## 4. Performance Guidelines

When using the Python API, you **must** wrap your execution block in PyTorch optimization contexts to achieve 30+ FPS:

```python
import torch

# ... initialization ...

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Call init_state() here
    while True:
        # Call track() here
```
Failure to include `inference_mode` and `autocast` will result in degraded performance and massive VRAM consumption.
