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

## the RSI Dataset

The **RSI Dataset** used in this work can be downloaded from the [Releases page](https://github.com/JasonBao05/HF-YOLO/releases/latest). The dataset is split into 3 parts (`rsi_dataset.zip.001-003`); download all parts and extract `rsi_dataset.zip.001` with 7-Zip to obtain the complete dataset.

## License

MIT
