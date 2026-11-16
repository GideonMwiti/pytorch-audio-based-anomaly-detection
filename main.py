#!/usr/bin/env python3
"""
Project 98: Audio-Based Anomaly Detection (Industrial Machines, ECG/PCG, etc.)
Unsupervised acoustic condition monitoring using deep convolutional autoencoders,
spectral reconstruction error, and latent space Mahalanobis outlier scoring.
"""

import os
import json
import time
import math
import random
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix

# ---------------------------------------------------------
# 1. Physical Acoustic Synthesis Engine (Normal vs Anomalous)
# ---------------------------------------------------------

SAMPLE_RATE = 16000
AUDIO_DURATION = 2.048  # 32768 samples
N_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
N_MELS = 64
N_FRAMES = 128  # 64 x 128 mel-spectrogram

def synth_normal_machine(dur: float = AUDIO_DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize normal rotating industrial machine audio (pumps, fans, motors).
    Steady motor hum (30/60/120 Hz), blade pass frequency (360 Hz), and broadband air flow.
    """
    n_pts = int(sr * dur)
    t = np.linspace(0, dur, n_pts, endpoint=False, dtype=np.float32)

    # Motor rotational fundamentals and electrical harmonics
    f_rot = random.uniform(28.0, 32.0)
    sig = 0.35 * np.sin(2.0 * np.pi * f_rot * t)
    sig += 0.25 * np.sin(2.0 * np.pi * (2 * f_rot) * t + 0.6)
    sig += 0.18 * np.sin(2.0 * np.pi * (4 * f_rot) * t + 1.2)

    # Fan blade pass frequency (12 blades * f_rot ~ 360 Hz)
    sig += 0.22 * np.sin(2.0 * np.pi * (12 * f_rot) * t + 0.9)
    sig += 0.12 * np.sin(2.0 * np.pi * (24 * f_rot) * t + 2.0)

    # Aerodynamic broadband air flow turbulence
    b_flow, a_flow = signal.butter(3, [80.0 / (sr / 2.0), 1600.0 / (sr / 2.0)], btype="band")
    flow_noise = signal.lfilter(b_flow, a_flow, np.random.normal(0.0, 0.12, n_pts).astype(np.float32))
    sig += flow_noise

    # Small background acoustic ambient hiss
    sig += np.random.normal(0.0, 0.02, n_pts).astype(np.float32)

    peak = np.max(np.abs(sig)) + 1e-6
    return (sig / peak * 0.90).astype(np.float32)

def synth_normal_heartbeat(dur: float = AUDIO_DURATION, bpm: float = 72.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize normal phonocardiogram (PCG) acoustic heart sounds.
    Regular S1 (mitral/tricuspid closure, ~50 Hz) and S2 (aortic/pulmonic closure, ~75 Hz).
    """
    n_pts = int(sr * dur)
    t = np.linspace(0, dur, n_pts, endpoint=False, dtype=np.float32)
    sig = np.zeros(n_pts, dtype=np.float32)

    beat_period = 60.0 / (bpm + random.uniform(-2.0, 2.0))
    cycle_samples = int(beat_period * sr)

    for start in range(0, n_pts, cycle_samples):
        # S1 valve closure (low-frequency resonant thud ~50 Hz, 100 ms)
        s1_len = int(0.10 * sr)
        if start + s1_len <= n_pts:
            t_s1 = np.linspace(0, 0.10, s1_len, dtype=np.float32)
            s1 = np.sin(2.0 * np.pi * 50.0 * t_s1) * np.sin(np.pi * t_s1 / 0.10)
            sig[start : start + s1_len] += 0.85 * s1

        # S2 valve closure (~0.30s systole later, ~75 Hz, 80 ms)
        s2_start = start + int(0.30 * sr)
        s2_len = int(0.08 * sr)
        if s2_start + s2_len <= n_pts:
            t_s2 = np.linspace(0, 0.08, s2_len, dtype=np.float32)
            s2 = np.sin(2.0 * np.pi * 75.0 * t_s2) * np.sin(np.pi * t_s2 / 0.08)
            sig[s2_start : s2_start + s2_len] += 0.65 * s2

    # Ambient body sound
    sig += np.random.normal(0.0, 0.025, n_pts).astype(np.float32)

    peak = np.max(np.abs(sig)) + 1e-6
    return (sig / peak * 0.90).astype(np.float32)

def synth_abnormal_sound(
    domain: str = "machine",
    anomaly_mode: str = "bearing_chatter",
    dur: float = AUDIO_DURATION,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Synthesize abnormal acoustic signals across industrial machine and physiological categories.
    """
    n_pts = int(sr * dur)
    t = np.linspace(0, dur, n_pts, endpoint=False, dtype=np.float32)

    if domain == "machine":
        sig = synth_normal_machine(dur, sr)

        if anomaly_mode == "bearing_chatter":
            # Periodic impulsive impact spikes at bearing fault frequency (BPFO ~ 85 Hz)
            impact_period = int(sr / 85.0)
            for idx in range(random.randint(0, 100), n_pts, impact_period):
                decay_len = min(220, n_pts - idx)
                decay_t = np.linspace(0, decay_len / sr, decay_len, dtype=np.float32)
                impact = np.exp(-decay_t * 600.0) * np.sin(2.0 * np.pi * 2600.0 * decay_t)
                sig[idx : idx + decay_len] += 0.55 * impact

        elif anomaly_mode == "cavitation_burst":
            # Violent high-frequency boiling/cavitation turbulence bursts (2500 - 6500 Hz)
            b_cav, a_cav = signal.butter(4, [2500.0 / (sr / 2.0), 6500.0 / (sr / 2.0)], btype="band")
            cav = signal.lfilter(b_cav, a_cav, np.random.normal(0.0, 0.45, n_pts).astype(np.float32))
            burst_env = np.clip(np.sin(2.0 * np.pi * 4.5 * t), 0.0, 1.0) ** 2
            sig += (cav * burst_env)

        elif anomaly_mode == "friction_screech":
            # Dry metal-on-metal abrasive friction squeal at high resonance (~3600 Hz)
            screech_f = 3600.0 + 350.0 * np.sin(2.0 * np.pi * 6.0 * t)
            screech = 0.48 * np.sin(2.0 * np.pi * screech_f * t)
            sig += screech

        elif anomaly_mode == "motor_unbalance":
            # Rotational unbalance / shaft misalignment: heavy subharmonic amplitude wobbling
            wobble = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.5 * t)
            sig = sig * wobble + 0.3 * np.sin(2.0 * np.pi * 15.0 * t)

    elif domain == "heart":
        sig = synth_normal_heartbeat(dur, sr=sr)

        if anomaly_mode == "heart_murmur":
            # Systolic murmur: turbulent high-frequency swishing between S1 and S2
            b_mur, a_mur = signal.butter(3, [250.0 / (sr / 2.0), 900.0 / (sr / 2.0)], btype="band")
            turb = signal.lfilter(b_mur, a_mur, np.random.normal(0.0, 0.35, n_pts).astype(np.float32))
            # Modulate between S1 and S2
            mur_env = np.clip(np.sin(2.0 * np.pi * 1.2 * t), 0.0, 1.0)
            sig += (turb * mur_env)

        elif anomaly_mode == "arrhythmia_extrasystole":
            # Premature irregular ventricular contraction interrupting normal rhythm
            extra_idx = int(0.65 * sr)
            extra_len = int(0.12 * sr)
            if extra_idx + extra_len <= n_pts:
                t_ex = np.linspace(0, 0.12, extra_len, dtype=np.float32)
                extra_beat = 0.95 * np.sin(2.0 * np.pi * 45.0 * t_ex) * np.sin(np.pi * t_ex / 0.12)
                sig[extra_idx : extra_idx + extra_len] += extra_beat

    peak = np.max(np.abs(sig)) + 1e-6
    return (sig / peak * 0.95).astype(np.float32)

# ---------------------------------------------------------
# 2. Spectrogram Feature Extraction Pipeline
# ---------------------------------------------------------

class AudioSpectrogramExtractor:
    """Extracts normalized log-mel spectrograms (64 mels x 128 frames) scaled to [0, 1]."""
    def __init__(self, sr: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = torch.hann_window(WIN_LENGTH)

        # Create Mel Filterbank
        mel_fb = self._create_mel_filterbank(sr, n_fft, n_mels)
        self.mel_fb = torch.tensor(mel_fb, dtype=torch.float32)

    def _create_mel_filterbank(self, sr, n_fft, n_mels):
        f_min, f_max = 20.0, sr / 2.0
        mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus = bin_pts[m - 1]
            f_m = bin_pts[m]
            f_m_plus = bin_pts[m + 1]
            for k in range(f_m_minus, f_m):
                fb[m - 1, k] = (k - bin_pts[m - 1]) / (bin_pts[m] - bin_pts[m - 1] + 1e-6)
            for k in range(f_m, f_m_plus):
                fb[m - 1, k] = (bin_pts[m + 1] - k) / (bin_pts[m + 1] - bin_pts[m] + 1e-6)
        return fb

    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, T) or (T,)
        Returns normalized log-mel spectrogram: (B, 1, 64, 128) in range [0, 1]
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        device = wav.device
        win = self.window.to(device)
        fb = self.mel_fb.to(device)

        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=WIN_LENGTH,
            window=win,
            return_complex=True
        )
        mag = torch.abs(stft)  # (B, n_fft//2 + 1, frames)
        mel = torch.matmul(fb, mag)  # (B, n_mels, frames)

        # Crop / pad to exact N_FRAMES (128)
        if mel.shape[-1] > N_FRAMES:
            mel = mel[:, :, :N_FRAMES]
        elif mel.shape[-1] < N_FRAMES:
            mel = F.pad(mel, (0, N_FRAMES - mel.shape[-1]))

        # Log compression and min-max normalization to [0, 1]
        log_mel = torch.log1p(10.0 * mel)
        norm_mel = log_mel / (log_mel.max() + 1e-6)
        return norm_mel.unsqueeze(1)

# ---------------------------------------------------------
# 3. Deep Convolutional Autoencoder Architecture
# ---------------------------------------------------------

class AudioAnomalyAutoencoder(nn.Module):
    """
    Deep Convolutional Autoencoder for Audio Anomaly Detection.
    Learns compact low-dimensional representations of normal operating sounds.
    """
    def __init__(self, in_c: int = 1, latent_dim: int = 48):
        super().__init__()
        # Encoder: (1, 64, 128) -> (128, 4, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, 16, 4, stride=2, padding=1),  # 32 x 64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),   # 16 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 8 x 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 4 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_enc = nn.Linear(128 * 4 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 8)

        # Decoder: (128, 4, 8) -> (1, 64, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 x 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 32 x 64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, in_c, 4, stride=2, padding=1), # 64 x 128
            nn.Sigmoid()  # Reconstructs normalized spectrogram in [0, 1]
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.fc_enc(h.view(h.size(0), -1))
        h_dec = self.fc_dec(z).view(h.size(0), 128, 4, 8)
        rec = self.decoder(h_dec)
        return rec, z

# ---------------------------------------------------------
# 4. Dataset Loader (Unsupervised Normal Training + Anomaly Benchmarking)
# ---------------------------------------------------------

class AcousticAnomalyDataset(Dataset):
    """
    Dataset for unsupervised anomaly detection.
    When is_train=True: returns only NORMAL operational audio.
    When is_train=False: returns balanced mix of NORMAL and ANOMALOUS audio across 6 failure modes.
    """
    def __init__(self, is_train: bool = True, n_normal: int = 240, n_anomaly: int = 120):
        self.is_train = is_train
        extractor = AudioSpectrogramExtractor()

        self.specs = []
        self.labels = []  # 0 = Normal, 1 = Anomaly
        self.types = []

        print(f"[INFO] Generating {'Training (Normal-Only)' if is_train else 'Evaluation (Normal + 6 Anomaly Types)'} Dataset...")

        # Normal samples (rotating machinery & resting PCG)
        for i in range(n_normal):
            if i % 2 == 0:
                wav = synth_normal_machine()
                stype = "normal_machine"
            else:
                wav = synth_normal_heartbeat()
                stype = "normal_heart"
            t_wav = torch.tensor(wav, dtype=torch.float32)
            spec = extractor.extract(t_wav).squeeze(0)
            self.specs.append(spec)
            self.labels.append(0)
            self.types.append(stype)

        # Anomalous samples (Evaluation only)
        if not is_train:
            anomaly_modes = [
                ("machine", "bearing_chatter"),
                ("machine", "cavitation_burst"),
                ("machine", "friction_screech"),
                ("machine", "motor_unbalance"),
                ("heart", "heart_murmur"),
                ("heart", "arrhythmia_extrasystole"),
            ]
            samples_per_mode = n_anomaly // len(anomaly_modes)
            for domain, mode in anomaly_modes:
                for _ in range(samples_per_mode):
                    wav = synth_abnormal_sound(domain=domain, anomaly_mode=mode)
                    t_wav = torch.tensor(wav, dtype=torch.float32)
                    spec = extractor.extract(t_wav).squeeze(0)
                    self.specs.append(spec)
                    self.labels.append(1)
                    self.types.append(mode)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx], self.types[idx]

# ---------------------------------------------------------
# 5. Model Training & Latent Calibration
# ---------------------------------------------------------

def train_anomaly_detector(epochs: int = 15, batch_size: int = 16, lr: float = 0.001, device: str = "cpu"):
    print("\n" + "=" * 65)
    print("  PROJECT 98: AUDIO-BASED ANOMALY DETECTION")
    print("=" * 65)
    print(f"Device: {device}")

    # Training dataset: 100% Normal samples (Unsupervised)
    train_dataset = AcousticAnomalyDataset(is_train=True, n_normal=240)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AudioAnomalyAutoencoder(in_c=1, latent_dim=48).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[INFO] Autoencoder Model Architecture: {sum(p.numel() for p in model.parameters()):,} parameters.")
    print(f"[INFO] Training unsupervised autoencoder for {epochs} epochs on normal data...\n")

    history = {"loss": [], "mse": [], "l1": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss, ep_mse, ep_l1 = 0.0, 0.0, 0.0

        for specs, _, _ in train_loader:
            specs = specs.to(device)
            rec, z = model(specs)

            loss_mse = F.mse_loss(rec, specs)
            loss_l1 = F.l1_loss(rec, specs)
            loss = loss_mse + 0.5 * loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_mse += loss_mse.item()
            ep_l1 += loss_l1.item()

        scheduler.step()
        n_batches = len(train_loader)
        ep_loss /= n_batches
        ep_mse /= n_batches
        ep_l1 /= n_batches

        history["loss"].append(ep_loss)
        history["mse"].append(ep_mse)
        history["l1"].append(ep_l1)

        ep_time = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | Loss: {ep_loss:.5f} | MSE: {ep_mse:.5f} | L1: {ep_l1:.5f}")

    train_duration = time.time() - start_time
    print(f"\n[INFO] Autoencoder trained in {train_duration:.1f}s.")

    # -----------------------------------------------------
    # Latent Normal Distribution Calibration (Mahalanobis)
    # -----------------------------------------------------
    print("[INFO] Calibrating Normal Latent Representation Statistics (Mean & Covariance)...")
    model.eval()
    latent_vectors = []
    recon_errors_normal = []

    with torch.no_grad():
        for specs, _, _ in train_loader:
            specs = specs.to(device)
            rec, z = model(specs)
            latent_vectors.append(z.cpu().numpy())
            mse = torch.mean((rec - specs) ** 2, dim=(1, 2, 3)).cpu().numpy()
            recon_errors_normal.extend(mse)

    latent_vectors = np.concatenate(latent_vectors, axis=0)  # (N, latent_dim)
    latent_mean = np.mean(latent_vectors, axis=0)
    latent_cov = np.cov(latent_vectors, rowvar=False) + 1e-4 * np.eye(latent_vectors.shape[1])
    latent_inv_cov = np.linalg.pinv(latent_cov)

    normal_recon_mean = float(np.mean(recon_errors_normal))
    normal_recon_std = float(np.std(recon_errors_normal) + 1e-6)

    diff_norm = latent_vectors - latent_mean
    normal_mahal_distances = np.sqrt(np.sum(np.dot(diff_norm, latent_inv_cov) * diff_norm, axis=1))
    normal_mahal_mean = float(np.mean(normal_mahal_distances))
    normal_mahal_std = float(np.std(normal_mahal_distances) + 1e-6)

    # Save model checkpoint
    save_path = "models/anomaly_autoencoder_best.pth"
    torch.save({
        "model_state": model.state_dict(),
        "latent_mean": latent_mean,
        "latent_inv_cov": latent_inv_cov,
        "normal_recon_mean": normal_recon_mean,
        "normal_recon_std": normal_recon_std,
        "normal_mahal_mean": normal_mahal_mean,
        "normal_mahal_std": normal_mahal_std,
        "history": history,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
    }, save_path)
    print(f"[INFO] Model checkpoint & latent statistics saved to '{save_path}'.")

    return model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std, history

# ---------------------------------------------------------
# 6. Evaluation, Benchmarking & Anomaly Scoring Suite
# ---------------------------------------------------------

def evaluate_and_benchmark(model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std, history, device: str = "cpu"):
    print("\n[INFO] Running Comprehensive Quantitative Evaluation on Multi-Fault Test Set...")
    model.eval()

    # Evaluation dataset: 120 Normal + 120 Anomalous (20 per each of the 6 fault modes)
    eval_dataset = AcousticAnomalyDataset(is_train=False, n_normal=120, n_anomaly=120)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    all_labels = []
    all_types = []
    recon_errors = []
    mahal_distances = []
    composite_scores = []
    latencies = []

    with torch.no_grad():
        for specs, labels, types in eval_loader:
            specs = specs.to(device)

            # Benchmark inference latency
            t0 = time.perf_counter()
            rec, z = model(specs)
            dt_ms = (time.perf_counter() - t0) * 1000.0 / specs.shape[0]
            latencies.extend([dt_ms] * specs.shape[0])

            # 1. Reconstruction MSE
            mse = torch.mean((rec - specs) ** 2, dim=(1, 2, 3)).cpu().numpy()
            recon_errors.extend(mse)

            # 2. Latent Mahalanobis Distance
            z_np = z.cpu().numpy()
            diff = z_np - latent_mean
            mahal = np.sqrt(np.sum(np.dot(diff, latent_inv_cov) * diff, axis=1))
            mahal_distances.extend(mahal)

            all_labels.extend(labels.numpy())
            all_types.extend(types)

    recon_errors = np.array(recon_errors)
    mahal_distances = np.array(mahal_distances)
    all_labels = np.array(all_labels)

    # Standardized composite score: 0.7 * Recon + 0.3 * Latent Outlier
    recon_z = (recon_errors - normal_recon_mean) / normal_recon_std
    mahal_z = (mahal_distances - normal_mahal_mean) / normal_mahal_std
    composite_scores = 0.7 * np.maximum(recon_z, 0.0) + 0.3 * np.maximum(mahal_z, 0.0)

    # Metrics computation
    roc_auc = float(roc_auc_score(all_labels, composite_scores))
    pr_auc = float(average_precision_score(all_labels, composite_scores))

    # Optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, composite_scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimal_threshold = float(thresholds[best_idx])

    # Predictions at optimal threshold
    predictions = (composite_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    precision = float(tp / (tp + fp + 1e-6))
    recall = float(tp / (tp + fn + 1e-6))
    specificity = float(tn / (tn + fp + 1e-6))
    f1_score = float(2.0 * precision * recall / (precision + recall + 1e-6))
    balanced_acc = float((recall + specificity) / 2.0)

    # Breakdown per anomaly mode
    type_recalls = {}
    unique_types = sorted(list(set(all_types)))
    for t_name in unique_types:
        if t_name not in ["normal_machine", "normal_heart"]:
            mask = (np.array(all_types) == t_name)
            type_recall = float(np.mean(predictions[mask] == 1))
            type_recalls[t_name] = round(type_recall, 4)

    mean_latency_ms = float(np.mean(latencies))
    rtf = (mean_latency_ms / 1000.0) / AUDIO_DURATION
    speedup = AUDIO_DURATION / (mean_latency_ms / 1000.0)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1_score, 4),
        "per_anomaly_recall": type_recalls,
        "mean_latency_ms": round(mean_latency_ms, 2),
        "real_time_factor_rtf": round(rtf, 5),
        "real_time_speedup_factor": round(speedup, 1),
    }

    # Save metrics JSON
    metrics_path = "outputs/anomaly_detection_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to '{metrics_path}'.")

    # Generate canonical demo audio samples
    sf.write("outputs/sample_normal_pump.wav", synth_normal_machine(), SAMPLE_RATE)
    sf.write("outputs/sample_bearing_fault.wav", synth_abnormal_sound("machine", "bearing_chatter"), SAMPLE_RATE)
    sf.write("outputs/sample_cavitation.wav", synth_abnormal_sound("machine", "cavitation_burst"), SAMPLE_RATE)
    sf.write("outputs/sample_friction_screech.wav", synth_abnormal_sound("machine", "friction_screech"), SAMPLE_RATE)
    sf.write("outputs/sample_normal_pcg.wav", synth_normal_heartbeat(), SAMPLE_RATE)
    sf.write("outputs/sample_heart_murmur.wav", synth_abnormal_sound("heart", "heart_murmur"), SAMPLE_RATE)
    print("[INFO] Canonical audio samples written to 'outputs/'.")

    # -----------------------------------------------------
    # Generate 6-Panel Diagnostic Dashboard
    # -----------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Waveforms Comparison (Normal Pump vs Bearing Fault)
    t_axis = np.linspace(0, AUDIO_DURATION, N_SAMPLES)
    w_norm = synth_normal_machine()
    w_anom = synth_abnormal_sound("machine", "bearing_chatter")
    ax1 = axes[0, 0]
    ax1.plot(t_axis, w_norm, color="#3498db", alpha=0.7, label="Normal Industrial Pump")
    ax1.plot(t_axis, w_anom, color="#e74c3c", alpha=0.6, label="Bearing Fault (Impact Spikes)")
    ax1.set_title("Acoustic Waveforms: Normal Pump vs Bearing Fault", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Input Spectrograms (Bearing Fault)
    extractor = AudioSpectrogramExtractor()
    spec_anom = extractor.extract(torch.tensor(w_anom, dtype=torch.float32)).to(device)
    with torch.no_grad():
        rec_anom, _ = model(spec_anom)
    residual_anom = torch.abs(spec_anom - rec_anom)

    ax2 = axes[0, 1]
    im2 = ax2.imshow(spec_anom[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title("Anomalous Spectrogram: Bearing Chatter with High-Frequency Shocks", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Mel Frequency Bins")
    fig.colorbar(im2, ax=ax2, label="Normalized Log-Mel")

    # Panel 3: Reconstruction Error Residual Map
    ax3 = axes[1, 0]
    im3 = ax3.imshow(residual_anom[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="hot")
    ax3.set_title("Reconstruction Residual Map (|Input - Reconstructed|)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("Mel Frequency Bins")
    fig.colorbar(im3, ax=ax3, label="Error Magnitude")

    # Panel 4: Score Distributions & Decision Threshold
    ax4 = axes[1, 1]
    norm_scores = composite_scores[all_labels == 0]
    anom_scores = composite_scores[all_labels == 1]
    ax4.hist(norm_scores, bins=25, alpha=0.65, color="#2ecc71", label=f"Normal Sounds (N={len(norm_scores)})", density=True)
    ax4.hist(anom_scores, bins=25, alpha=0.65, color="#e74c3c", label=f"Anomalies (N={len(anom_scores)})", density=True)
    ax4.axvline(optimal_threshold, color="#2c3e50", linestyle="--", linewidth=2.0, label=f"Threshold tau* = {optimal_threshold:.2f}")
    ax4.set_title(f"Composite Anomaly Score Separation (ROC-AUC: {roc_auc:.4f})", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Composite Anomaly Score")
    ax4.set_ylabel("Probability Density")
    ax4.legend(loc="upper right")
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Panel 5: ROC & PR Curves
    ax5 = axes[2, 0]
    prec_pts, rec_pts, _ = precision_recall_curve(all_labels, composite_scores)
    ax5.plot(fpr, tpr, color="#3498db", linewidth=2.0, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax5.plot(rec_pts, prec_pts, color="#e67e22", linewidth=2.0, linestyle="--", label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax5.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax5.set_title("Detection Performance: ROC and Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax5.set_xlabel("False Positive Rate / Recall")
    ax5.set_ylabel("True Positive Rate / Precision")
    ax5.legend(loc="lower right")
    ax5.grid(True, linestyle="--", alpha=0.5)

    # Panel 6: Per-Fault Mode Detection Sensitivity
    ax6 = axes[2, 1]
    fault_names = list(type_recalls.keys())
    fault_scores = [type_recalls[k] * 100.0 for k in fault_names]
    bar_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#1abc9c", "#f39c12", "#d35400"]
    bars = ax6.barh(fault_names, fault_scores, color=bar_colors[:len(fault_names)], alpha=0.85)
    ax6.set_xlim(0, 105)
    ax6.set_title("Per-Fault Mode Anomaly Detection Sensitivity", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Detection Recall (%)")
    for bar in bars:
        w = bar.get_width()
        ax6.text(w + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax6.grid(True, linestyle="--", alpha=0.5, axis="x")

    report_path = "outputs/anomaly_detection_report.png"
    plt.savefig(report_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Diagnostic report dashboard saved to '{report_path}'.")

    # Print summary table
    print("\n" + "=" * 65)
    print("      AUDIO ANOMALY DETECTION BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"ROC-AUC:                {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:                 {metrics['pr_auc']:.4f}")
    print(f"Optimal Threshold (tau*):{metrics['optimal_threshold']:.4f}")
    print(f"F1-Score:               {metrics['f1_score']:.4f}")
    print(f"Sensitivity (Recall):   {metrics['recall']:.4f}")
    print(f"Specificity:            {metrics['specificity']:.4f}")
    print(f"Balanced Accuracy:      {metrics['balanced_accuracy']:.4f}")
    print(f"Mean Inference Latency: {metrics['mean_latency_ms']:.2f} ms")
    print(f"Real-Time Speedup:      {metrics['real_time_speedup_factor']:.1f}x real-time (RTF: {metrics['real_time_factor_rtf']:.5f})")
    print("-" * 65)
    print("Per-Anomaly Type Recall Breakdown:")
    for k, v in metrics["per_anomaly_recall"].items():
        print(f"  - {k:<25}: {v * 100.0:.1f}%")
    print("=" * 65 + "\n")

    return metrics

# ---------------------------------------------------------
# 7. Main Entrypoint
# ---------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std, history = train_anomaly_detector(
        epochs=15,
        batch_size=16,
        lr=0.001,
        device=device
    )
    evaluate_and_benchmark(model, latent_mean, latent_inv_cov, normal_recon_mean, normal_recon_std, normal_mahal_mean, normal_mahal_std, history, device=device)

if __name__ == "__main__":
    main()

