"""
HF-YOLO Frequency Analysis and Visualization
===========================================
Experiment 1: Feature Activation Heatmap
Experiment 2: Post-training Spectral Analysis

Run:
python FALAexp.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

# ================================================================
# Configuration
# ================================================================

# sys.path already set

WEIGHT_YOLO11 = '/root/autodl-tmp/hfyolo/yolo11_exp5.pt'
WEIGHT_MLLA   = '/root/autodl-tmp/hfyolo/mlla_exp36.pt'
WEIGHT_FALA   = '/root/autodl-tmp/hfyolo/hfyolo_seed0.pt'

TEST_IMAGE = '/root/autodl-tmp/hfyolo/DJI_20231018113710_0023_W.jpeg'


# ================================================================
# Utility Functions
# ================================================================

def load_image_keep_ratio(img_path, max_size=1344):
    """Load image while preserving aspect ratio"""
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    scale = max_size / max(orig_h, orig_w)
    new_w = (int(orig_w * scale) // 32) * 32
    new_h = (int(orig_h * scale) // 32) * 32

    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    print(f"Original: {orig_w}x{orig_h} -> Resized: {new_w}x{new_h}")

    return img_rgb, img_resized, tensor, orig_h, orig_w, new_h, new_w


def radial_energy(feat):
    """Compute radial spectral energy distribution"""
    B, C, H, W = feat.shape

    fft = torch.fft.fft2(feat.float())
    mag = torch.abs(fft).mean(dim=(0, 1))
    mag = torch.fft.fftshift(mag)

    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = ((y - cy).float() ** 2 + (x - cx).float() ** 2).sqrt()

    energy = []

    for r in range(max_r):
        mask = (dist >= r) & (dist < r + 1)
        energy.append(mag[mask].mean().item() if mask.sum() > 0 else 0)

    return np.array(energy)


# ================================================================
# Experiment 1: Feature Activation Heatmap
# ================================================================

def run_experiment_1():
    """Feature activation heatmap"""
    print("\n" + "=" * 60)
    print("Experiment 1: Feature Activation Heatmap")
    print("=" * 60)

    from ultralytics import YOLO

    img_rgb, img_resized, img_tensor, orig_h, orig_w, new_h, new_w = load_image_keep_ratio(TEST_IMAGE)

    configs = [
        ('YOLO11 (CNN)', WEIGHT_YOLO11),
        ('MLLA', WEIGHT_MLLA),
        ('FALA (Ours)', WEIGHT_FALA),
    ]

    layer_idx = None

    # Fixed 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    # Input image
    axes[0].imshow(img_resized)
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')

    for idx, (name, weight) in enumerate(configs):
        print(f"\nLoading model: {name}")
        model = YOLO(weight)

        if layer_idx is None:
            print("  Model layers:")
            for i, layer in enumerate(model.model.model):
                cls_name = layer.__class__.__name__
                print(f"    Layer {i}: {cls_name}")

                if layer_idx is None and any(
                    kw in cls_name for kw in ['CSP_FALA', 'C3k2', 'C3kMLLA', 'MLLA']
                ):
                    layer_idx = i

            if layer_idx is None:
                layer_idx = 4

            print(f"  -> Hook layer: {layer_idx}")

        features = {}

        def hook_fn(m, inp, out):
            features['out'] = out.detach().cpu()

        handle = model.model.model[layer_idx].register_forward_hook(hook_fn)

        device = next(model.model.parameters()).device

        with torch.no_grad():
            model.model(img_tensor.to(device))

        handle.remove()

        feat = features['out']
        print(f"  Feature map: {feat.shape}")

        activation = feat[0].mean(dim=0).numpy()

        low = np.percentile(activation, 2)
        high = np.percentile(activation, 98)

        activation = np.clip((activation - low) / (high - low + 1e-8), 0, 1)

        activation_up = cv2.resize(
            activation,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        axes[idx + 1].imshow(img_resized)
        axes[idx + 1].imshow(
            activation_up,
            alpha=0.5,
            cmap='jet',
            vmin=0,
            vmax=1
        )

        axes[idx + 1].set_title(name, fontsize=14)
        axes[idx + 1].axis('off')

        del model
        torch.cuda.empty_cache()

    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.04,
        wspace=0.04,
        hspace=0.12
    )

    plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved: attention_heatmap.png")


# ================================================================
# Experiment 2: Post-training Spectral Analysis
# ================================================================

def run_experiment_2():
    """Post-training spectral comparison"""
    print("\n" + "=" * 60)
    print("Experiment 2: Post-training Spectral Analysis")
    print("=" * 60)

    from ultralytics import YOLO

    img_rgb, img_resized, img_tensor, orig_h, orig_w, new_h, new_w = load_image_keep_ratio(TEST_IMAGE)

    configs = [
        ('YOLO11 (CNN)', WEIGHT_YOLO11),
        ('MLLA', WEIGHT_MLLA),
        ('FALA (Ours)', WEIGHT_FALA),
    ]

    colors = {
        'YOLO11 (CNN)': '#888780',
        'MLLA': '#185FA5',
        'FALA (Ours)': '#E24B4A'
    }

    layer_idx = None
    results = {}

    for name, weight in configs:
        print(f"\nLoading model: {name}")
        model = YOLO(weight)

        if layer_idx is None:
            for i, layer in enumerate(model.model.model):
                cls_name = layer.__class__.__name__
                if any(kw in cls_name for kw in ['CSP_FALA', 'C3k2', 'C3kMLLA', 'MLLA']):
                    layer_idx = i

            if layer_idx is None:
                layer_idx = 4

        features = {}

        def hook_fn(m, inp, out):
            features['out'] = out.detach().cpu()

        handle = model.model.model[layer_idx].register_forward_hook(hook_fn)

        device = next(model.model.parameters()).device

        with torch.no_grad():
            model.model(img_tensor.to(device))

        handle.remove()

        feat = features['out']
        print(f"  Feature map: {feat.shape}")

        results[name] = radial_energy(feat)

        del model
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]

    for name, energy in results.items():
        freqs = np.arange(len(energy))
        energy_norm = energy / energy.sum()

        ax.semilogy(
            freqs,
            energy_norm + 1e-10,
            label=name,
            color=colors[name],
            linewidth=1.5
        )

    mid = len(freqs) // 2

    ax.axvspan(mid, len(freqs), alpha=0.08, color='red')
    ax.set_xlabel('Frequency (low -> high)')
    ax.set_ylabel('Normalized energy (log scale)')
    ax.set_title('(a) Full spectrum')
    ax.legend()

    ax = axes[1]

    for name, energy in results.items():
        freqs = np.arange(len(energy))
        energy_norm = energy / energy.sum()

        ax.plot(
            freqs[mid:],
            energy_norm[mid:],
            label=name,
            color=colors[name],
            linewidth=1.5
        )

    ax.set_xlabel('High-frequency region')
    ax.set_ylabel('Normalized energy')
    ax.set_title('(b) High-frequency response')
    ax.legend()

    # 在高频区域标注能量比
    hf_ratios = {}
    for name, energy in results.items():
        total = energy.sum()
        mid = len(energy) // 2
        hf = energy[mid:].sum()
        hf_ratios[name] = hf / total * 100

    # 标注在右图（高频区域）
    y_positions = [0.85, 0.75, 0.65]
    text_colors = {'YOLO11 (CNN)': '#888780', 'MLLA': '#185FA5', 'FALA (Ours)': '#E24B4A'}
    for i, (name, ratio) in enumerate(hf_ratios.items()):
        axes[1].text(0.98, y_positions[i], f'{name}: {ratio:.1f}%',
                    transform=axes[1].transAxes,
                    ha='right', va='top',
                    color=text_colors.get(name, 'black'),
                    fontsize=10, fontweight='bold')

    # 加注释说明
    axes[1].text(0.02, 0.05, 'Higher = more HF preserved',
                transform=axes[1].transAxes,
                ha='left', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('freq_spectrum_trained.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved: freq_spectrum_trained.png")

    print("\nHigh-frequency energy ratio:")

    for name, energy in results.items():
        total = energy.sum()
        hf = energy[mid:].sum()
        print(f"  {name}: {hf / total * 100:.1f}%")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("HF-YOLO Frequency Analysis and Visualization")
    print("=" * 60)

    has_image = Path(TEST_IMAGE).exists()
    has_weights = all(Path(w).exists() for w in [
        WEIGHT_YOLO11,
        WEIGHT_MLLA,
        WEIGHT_FALA
    ])

    if has_image and has_weights:
        run_experiment_1()
        run_experiment_2()
    else:
        if not has_image:
            print(f"\nImage not found: {TEST_IMAGE}")

        if not has_weights:
            print("Missing model weights.")

    print("\n" + "=" * 60)
    print("Completed.")
    print("Generated:")
    print("  attention_heatmap.png")
    print("  freq_spectrum_trained.png")
    print("=" * 60)