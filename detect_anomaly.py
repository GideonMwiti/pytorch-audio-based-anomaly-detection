#!/usr/bin/env python3
"""
Production CLI for Project 98: Audio-Based Anomaly Detection
Classifies audio recordings as Normal vs. Anomalous using spectral reconstruction
error and latent Mahalanobis distance scoring.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from main import (
    AudioAnomalyAutoencoder,
    AudioSpectrogramExtractor,
    SAMPLE_RATE,
    AUDIO_DURATION,
    N_SAMPLES,
    synth_normal_machine,
    synth_normal_heartbeat,
    synth_abnormal_sound,
)

def load_anomaly_detector(checkpoint_path: str = "models/anomaly_autoencoder_best.pth", device: str = "cpu"):
    """Load pre-trained model and calibrated normal statistics."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please run 'python main.py' first to train the autoencoder."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AudioAnomalyAutoencoder(in_c=1, latent_dim=48).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    latent_mean = ckpt["latent_mean"]
    latent_inv_cov = ckpt["latent_inv_cov"]
    normal_recon_mean = ckpt["normal_recon_mean"]
    normal_recon_std = ckpt["normal_recon_std"]
    normal_mahal_mean = ckpt.get("normal_mahal_mean", 5.0)
    normal_mahal_std = ckpt.get("normal_mahal_std", 2.0)

    return model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std

def predict_audio_anomaly(
    wav_path: str,
    model,
    latent_mean,
    latent_inv_cov,
    normal_recon_mean,
    normal_recon_std,
    normal_mahal_mean: float = 5.0,
    normal_mahal_std: float = 2.0,
    threshold: float = 1.0,
    visualize: bool = False,
    output_plot: str = None,
    device: str = "cpu"
):
    """Run anomaly detection inference on a given audio file."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: '{wav_path}'")

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Pad or truncate to standard length
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    elif len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]

    t_wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    extractor = AudioSpectrogramExtractor()
    spec = extractor.extract(t_wav).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        rec, z = model(spec)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Reconstruction MSE
    mse = float(torch.mean((rec - spec) ** 2).item())

    # 2. Latent Mahalanobis Distance
    z_np = z.cpu().numpy()
    diff = z_np - latent_mean
    mahal = float(np.sqrt(np.sum(np.dot(diff, latent_inv_cov) * diff, axis=1))[0])

    # Standardized composite score
    recon_z = (mse - normal_recon_mean) / normal_recon_std
    mahal_z = (mahal - normal_mahal_mean) / normal_mahal_std
    composite_score = float(0.7 * max(0.0, recon_z) + 0.3 * max(0.0, mahal_z))

    is_anomaly = (composite_score >= threshold)
    confidence = min(100.0, max(50.0, 50.0 + abs(composite_score - threshold) * 25.0))

    status = "🚨 ANOMALY DETECTED" if is_anomaly else "✅ NORMAL OPERATION"

    print("\n" + "=" * 65)
    print("        AUDIO ANOMALY DETECTION REPORT")
    print("=" * 65)
    print(f"File:                   {wav_path}")
    print(f"Verdict:                {status}")
    print(f"Composite Anomaly Score:{composite_score:.4f} (Threshold: {threshold:.4f})")
    print(f"Reconstruction Error:   {mse:.6f} (Normal Baseline: {normal_recon_mean:.6f})")
    print(f"Latent Outlier Distance:{mahal:.4f}")
    print(f"Confidence Level:       {confidence:.1f}%")
    print(f"Inference Latency:      {dt_ms:.2f} ms ({AUDIO_DURATION / (dt_ms / 1000.0):.1f}x real-time)")
    print("=" * 65)

    if visualize:
        out_fig = output_plot or f"{os.path.splitext(wav_path)[0]}_anomaly_analysis.png"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(spec[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Input Spectrogram", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("Mel Bins")

        axes[1].imshow(rec[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title("Autoencoder Reconstruction", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Bins")

        residual = torch.abs(spec - rec)[0, 0].cpu().numpy()
        im = axes[2].imshow(residual, aspect="auto", origin="lower", cmap="hot")
        axes[2].set_title(f"Reconstruction Error Map (Score: {composite_score:.2f})", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Time Frames")
        axes[2].set_ylabel("Mel Bins")
        fig.colorbar(im, ax=axes[2], label="Error")

        plt.tight_layout()
        plt.savefig(out_fig, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Anomaly visual report saved to '{out_fig}'.")

    return {
        "is_anomaly": is_anomaly,
        "composite_score": composite_score,
        "reconstruction_error": mse,
        "mahalanobis_distance": mahal,
        "latency_ms": dt_ms,
    }

def run_demo(checkpoint_path: str = "models/anomaly_autoencoder_best.pth", device: str = "cpu"):
    """Run built-in multi-fault demonstration suite."""
    print("\n[DEMO] Initializing Demo Test Audio Samples across Industrial and Bioacoustic Domains...")

    model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std = load_anomaly_detector(checkpoint_path, device)

    # Read optimal threshold from metrics if available
    threshold = 1.0
    metrics_path = "outputs/anomaly_detection_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            threshold = data.get("optimal_threshold", 1.0)

    demo_cases = [
        ("Normal Industrial Pump", "outputs/sample_normal_pump.wav", synth_normal_machine()),
        ("Bearing Fault (Chatter Spikes)", "outputs/sample_bearing_fault.wav", synth_abnormal_sound("machine", "bearing_chatter")),
        ("Pump Cavitation Burst", "outputs/sample_cavitation.wav", synth_abnormal_sound("machine", "cavitation_burst")),
        ("Dry Friction Screech", "outputs/sample_friction_screech.wav", synth_abnormal_sound("machine", "friction_screech")),
        ("Normal Phonocardiogram (PCG)", "outputs/sample_normal_pcg.wav", synth_normal_heartbeat()),
        ("Heart Murmur (Turbulent Swish)", "outputs/sample_heart_murmur.wav", synth_abnormal_sound("heart", "heart_murmur")),
    ]

    os.makedirs("outputs", exist_ok=True)
    for title, wav_path, wav_data in demo_cases:
        sf.write(wav_path, wav_data, SAMPLE_RATE)
        print(f"\n>>> Testing Scenario: {title}")
        predict_audio_anomaly(
            wav_path=wav_path,
            model=model,
            latent_mean=latent_mean,
            latent_inv_cov=latent_inv_cov,
            normal_recon_mean=normal_recon_mean,
            normal_recon_std=normal_recon_std,
            normal_mahal_mean=normal_mahal_mean,
            normal_mahal_std=normal_mahal_std,
            threshold=threshold,
            visualize=False,
            device=device
        )

def main():
    parser = argparse.ArgumentParser(description="Audio-Based Anomaly Detection CLI")
    parser.add_argument("--input", type=str, default=None, help="Path to input audio file (.wav)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (default: auto from training)")
    parser.add_argument("--visualize", action="store_true", help="Generate localized spectrogram error plot")
    parser.add_argument("--demo", action="store_true", help="Run comprehensive multi-fault demonstration")
    parser.add_argument("--checkpoint", type=str, default="models/anomaly_autoencoder_best.pth", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.demo:
        run_demo(checkpoint_path=args.checkpoint, device=args.device)
        return

    if args.input is None:
        print("[ERROR] Please provide an audio file via --input <file.wav> or run --demo.")
        parser.print_help()
        sys.exit(1)

    model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std = load_anomaly_detector(args.checkpoint, args.device)

    threshold = args.threshold
    if threshold is None:
        metrics_path = "outputs/anomaly_detection_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                threshold = data.get("optimal_threshold", 1.0)
        else:
            threshold = 1.0

    predict_audio_anomaly(
        wav_path=args.input,
        model=model,
        latent_mean=latent_mean,
        latent_inv_cov=latent_inv_cov,
        normal_recon_mean=normal_recon_mean,
        normal_recon_std=normal_recon_std,
        normal_mahal_mean=normal_mahal_mean,
        normal_mahal_std=normal_mahal_std,
        threshold=threshold,
        visualize=args.visualize,
        device=args.device
    )

if __name__ == "__main__":
    main()

