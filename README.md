# HF-YOLO

Frequency-Aware Linear Attention for Small Object Detection in UAV Remote Sensing of River and Lake Shoreline Areas.

## Files

| File | Description |
|------|-------------|
| `CSP_FALA.py` | Frequency-Aware Linear Attention (FALA) module, replaces C3K2 in YOLO11 backbone and neck |
| `HAT.py` | Hybrid Attention Transformer (HAT) module for small target super-resolution enhancement |
| `hf-yolo-obb.yaml` | YOLO model configuration file defining the HF-YOLO network architecture |
| `FALAexp.py` | Visualization experiments: attention heatmap and frequency spectrum analysis |

## Usage

Copy `CSP_FALA.py` and `HAT.py` to `ultralytics/nn/Extramodules/

## License

MIT
