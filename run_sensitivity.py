#!/usr/bin/env python3
"""
Run one-parameter-at-a-time sensitivity experiments.

Example:
    /opt/anaconda3/envs/speech311/bin/python run_sensitivity.py \
        --config configs/representation.yaml

The script writes:
    raw_results.csv
    summary_by_value.csv
    change_efficiency.csv
    recommendations.md
    plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tarfile
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "speech_sensitivity_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "speech_sensitivity_matplotlib_cache"))

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import whisper
import yaml


DEFAULT_DATA_ROOT = Path("/Users/home2/Desktop/xy/year1 /summer project")
DEFAULT_TARBALL = DEFAULT_DATA_ROOT / "speech_commands_v0.02.tar.gz"


@dataclass(frozen=True)
class ExperimentConfig:
    sample_rate: int = 16000
    frame_size_ms: float = 25.0
    hop_ms: float = 10.0
    n_mels: int = 80
    fmin_hz: float = 0.0
    fmax_hz: float = 8000.0
    representation_type: str = "mel"
    n_mfcc: int = 20
    mel_scale: str = "linear"
    normalization_method: str = "none"
    dynamic_range_db: float | None = None
    quant_bits: int = 0
    time_downsample_factor: int = 1
    time_downsample_method: str = "none"
    frequency_mask: str = "none"
    snr_db: float | None = None
    noise_type: str = "white"
    reconstruction_method: str = "griffin_lim"
    griffin_lim_iters: int = 32
    whisper_model: str = "tiny"

    @property
    def win_length(self) -> int:
        return max(2, int(round(self.sample_rate * self.frame_size_ms / 1000.0)))

    @property
    def hop_length(self) -> int:
        return max(1, int(round(self.sample_rate * self.hop_ms / 1000.0)))

    @property
    def n_fft(self) -> int:
        return 1 << (self.win_length - 1).bit_length()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dataset(data_root: Path, tarball: Path, labels: Iterable[str]) -> None:
    if all((data_root / label).exists() for label in labels):
        return
    if not tarball.exists():
        raise FileNotFoundError(f"Missing extracted labels and tarball: {tarball}")
    with tarfile.open(tarball, "r:gz") as archive:
        archive.extractall(data_root)


def select_audio_files(
    data_root: Path,
    labels: Iterable[str],
    files_per_label: int | None,
    max_files: int | None,
    selection: str,
    seed: int,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    selected: list[Path] = []
    for label in labels:
        files = sorted((data_root / label).glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No WAV files found for label '{label}'")
        if selection == "random":
            files = [files[i] for i in rng.permutation(len(files))]
        selected.extend(files[:files_per_label] if files_per_label is not None else files)
    return selected[:max_files] if max_files is not None else selected


def apply_config_updates(cfg: ExperimentConfig, updates: dict[str, Any]) -> ExperimentConfig:
    clean = {k: normalize_value(v) for k, v in updates.items()}
    return replace(cfg, **clean)


def normalize_value(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in {"null"}:
        return None
    return value


def parameter_value(cfg: ExperimentConfig, name: str) -> Any:
    return getattr(cfg, name)


def with_parameter(base: ExperimentConfig, name: str, value: Any) -> ExperimentConfig:
    return replace(base, **{name: normalize_value(value)})


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def colored_noise(kind: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    if kind == "white":
        return white
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white))
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    if kind == "pink":
        spectrum = spectrum / np.sqrt(freqs)
    elif kind == "brown":
        spectrum = spectrum / freqs
    else:
        raise ValueError(f"Unsupported noise_type: {kind}")
    noise = np.fft.irfft(spectrum, n=len(white)).astype(np.float32)
    return noise / max(float(np.std(noise)), 1e-12)


def add_noise(audio: np.ndarray, cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.snr_db is None:
        return audio
    signal_power = float(np.mean(audio**2) + 1e-12)
    noise_power = signal_power / (10.0 ** (float(cfg.snr_db) / 10.0))
    noise = colored_noise(cfg.noise_type, audio.shape, rng)
    noise = noise * math.sqrt(noise_power / max(float(np.mean(noise**2)), 1e-12))
    return np.clip(audio + noise, -1.0, 1.0)


def mel_spectrogram(audio: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hann",
        center=True,
        power=1.0,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin_hz,
        fmax=cfg.fmax_hz,
    ).astype(np.float32)


def apply_frequency_mask(mel: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    mask = cfg.frequency_mask
    if mask in {"none", None}:
        return mel
    freqs = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin_hz, fmax=cfg.fmax_hz)
    keep = np.ones_like(freqs, dtype=bool)
    if mask.startswith("low_below_"):
        keep &= freqs >= float(mask.split("_")[-1])
    elif mask.startswith("high_above_"):
        keep &= freqs <= float(mask.split("_")[-1])
    elif mask.startswith("band_"):
        _, lo, hi = mask.split("_")
        keep &= ~((freqs >= float(lo)) & (freqs <= float(hi)))
    else:
        raise ValueError(f"Unsupported frequency_mask: {mask}")
    out = mel.copy()
    out[~keep, :] = 0.0
    return out


def apply_dynamic_range(mel: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    if cfg.dynamic_range_db is None:
        return mel
    peak = float(np.max(mel))
    floor = peak * (10.0 ** (-float(cfg.dynamic_range_db) / 20.0))
    return np.maximum(mel, floor).astype(np.float32)


def time_downsample(rep: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, dict[str, Any]]:
    factor = int(cfg.time_downsample_factor)
    method = cfg.time_downsample_method
    if factor <= 1 or method == "none":
        return rep, {"original_frames": rep.shape[1], "factor": 1, "method": "none"}
    frames = rep.shape[1]
    if method == "decimate":
        down = rep[:, ::factor]
    elif method == "mean":
        trim = frames - (frames % factor)
        body = rep[:, :trim].reshape(rep.shape[0], -1, factor).mean(axis=2)
        down = np.concatenate([body, rep[:, trim:]], axis=1) if trim < frames else body
    elif method == "linear":
        x_old = np.arange(frames)
        x_new = np.linspace(0, frames - 1, max(1, math.ceil(frames / factor)))
        down = np.vstack([np.interp(x_new, x_old, row) for row in rep]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported time_downsample_method: {method}")
    return down.astype(np.float32), {"original_frames": frames, "factor": factor, "method": method}


def time_upsample(rep: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    frames = int(meta["original_frames"])
    if meta["factor"] <= 1 or meta["method"] == "none":
        return rep[:, :frames]
    x_old = np.linspace(0, frames - 1, rep.shape[1])
    x_new = np.arange(frames)
    return np.vstack([np.interp(x_new, x_old, row) for row in rep]).astype(np.float32)


def normalize_rep(rep: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, float]]:
    if method == "none":
        return rep, {}
    if method == "per_file_minmax":
        lo, hi = float(rep.min()), float(rep.max())
        return ((rep - lo) / max(hi - lo, 1e-12)).astype(np.float32), {"lo": lo, "hi": hi}
    if method == "per_file_zscore":
        mean, std = float(rep.mean()), float(rep.std())
        return ((rep - mean) / max(std, 1e-12)).astype(np.float32), {"mean": mean, "std": std}
    raise ValueError(f"Unsupported normalization_method: {method}")


def denormalize_rep(rep: np.ndarray, method: str, meta: dict[str, float]) -> np.ndarray:
    if method == "none":
        return rep
    if method == "per_file_minmax":
        return rep * (meta["hi"] - meta["lo"]) + meta["lo"]
    if method == "per_file_zscore":
        return rep * meta["std"] + meta["mean"]
    raise ValueError(f"Unsupported normalization_method: {method}")


def quantize(rep: np.ndarray, bits: int) -> tuple[np.ndarray, dict[str, float | None]]:
    if bits <= 0:
        return rep, {"quant_min": None, "quant_max": None}
    levels = (1 << int(bits)) - 1
    lo, hi = float(rep.min()), float(rep.max())
    if hi <= lo:
        return rep.copy(), {"quant_min": lo, "quant_max": hi}
    q = np.round((rep - lo) / (hi - lo) * levels)
    restored = q / levels * (hi - lo) + lo
    return restored.astype(np.float32), {"quant_min": lo, "quant_max": hi}


def encode_representation(audio: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    mel = mel_spectrogram(audio, cfg)
    mel = apply_frequency_mask(mel, cfg)
    mel = apply_dynamic_range(mel, cfg)
    source_mel = mel.copy()

    if cfg.representation_type == "mel":
        rep = mel
    elif cfg.representation_type == "mfcc":
        mel_db = librosa.amplitude_to_db(np.maximum(mel, 1e-10), ref=1.0)
        rep = librosa.feature.mfcc(S=mel_db, n_mfcc=cfg.n_mfcc).astype(np.float32)
    else:
        raise ValueError(f"Unsupported representation_type: {cfg.representation_type}")

    if cfg.mel_scale == "log":
        if cfg.representation_type != "mel":
            raise ValueError("mel_scale='log' is only supported for representation_type='mel'")
        rep = np.log1p(np.maximum(rep, 0.0)).astype(np.float32)
    elif cfg.mel_scale != "linear":
        raise ValueError(f"Unsupported mel_scale: {cfg.mel_scale}")

    rep, time_meta = time_downsample(rep, cfg)
    rep, norm_meta = normalize_rep(rep, cfg.normalization_method)
    compressed, quant_meta = quantize(rep, int(cfg.quant_bits))
    meta = {"time": time_meta, "normalization": norm_meta, "quantization": quant_meta}
    return compressed, meta, source_mel


def decode_to_mel(rep: np.ndarray, meta: dict[str, Any], cfg: ExperimentConfig) -> np.ndarray:
    rep = denormalize_rep(rep, cfg.normalization_method, meta["normalization"])
    rep = time_upsample(rep, meta["time"])
    if cfg.mel_scale == "log":
        rep = np.expm1(rep).astype(np.float32)

    if cfg.representation_type == "mel":
        return np.maximum(rep, 0.0).astype(np.float32)
    if cfg.representation_type == "mfcc":
        mel = librosa.feature.inverse.mfcc_to_mel(
            rep,
            n_mels=cfg.n_mels,
            dct_type=2,
            norm="ortho",
            ref=1.0,
        )
        return np.maximum(mel, 0.0).astype(np.float32)
    raise ValueError(f"Unsupported representation_type: {cfg.representation_type}")


def reconstruct_audio(mel: np.ndarray, cfg: ExperimentConfig, length: int) -> np.ndarray:
    if cfg.reconstruction_method == "griffin_lim":
        audio = librosa.feature.inverse.mel_to_audio(
            M=np.maximum(mel, 0.0),
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window="hann",
            center=True,
            power=1.0,
            n_iter=cfg.griffin_lim_iters,
            fmin=cfg.fmin_hz,
            fmax=cfg.fmax_hz,
        )
    elif cfg.reconstruction_method == "zero_phase":
        stft_mag = librosa.feature.inverse.mel_to_stft(
            M=np.maximum(mel, 0.0),
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            power=1.0,
            fmin=cfg.fmin_hz,
            fmax=cfg.fmax_hz,
        )
        audio = librosa.istft(
            stft_mag.astype(np.complex64),
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window="hann",
            center=True,
            length=length,
        )
    else:
        raise ValueError(f"Unsupported reconstruction_method: {cfg.reconstruction_method}")
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return np.clip(audio[:length].astype(np.float32), -1.0, 1.0)


def compression_rate(original_audio: np.ndarray, rep: np.ndarray, cfg: ExperimentConfig) -> float:
    original_bytes = len(original_audio) * 2
    if cfg.quant_bits > 0:
        compressed_bytes = math.ceil(rep.size * int(cfg.quant_bits) / 8)
    else:
        compressed_bytes = rep.size * 4
    return float(original_bytes / max(compressed_bytes, 1))


def waveform_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    n = min(len(reference), len(estimate))
    err = reference[:n] - estimate[:n]
    return float(10.0 * np.log10((np.sum(reference[:n] ** 2) + 1e-12) / (np.sum(err**2) + 1e-12)))


def normalized_words(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return [word for word in text.split() if word]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(ref) + 1)
    dp[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[len(ref), len(hyp)] / len(ref))


def command_accuracy(label: str, transcript: str) -> float:
    return 1.0 if label.lower() in normalized_words(transcript) else 0.0


def transcribe_audio(model, audio: np.ndarray) -> str:
    result = model.transcribe(audio.astype(np.float32), fp16=False, language="en", verbose=False)
    return str(result.get("text", "")).strip()


def value_key(value: Any) -> str:
    return "null" if value is None else str(value)


def build_parameter_runs(config: dict[str, Any], base_cfg: ExperimentConfig) -> list[dict[str, Any]]:
    runs = []
    for spec in config["parameters"]:
        name = spec["name"]
        param_base = apply_config_updates(base_cfg, spec.get("fixed_overrides", {}))
        baseline_value = parameter_value(param_base, name)
        values = [normalize_value(v) for v in spec["values"]]
        if config["experiment"].get("include_baseline", True) and baseline_value not in values:
            values = [baseline_value] + values
        runs.append(
            {
                "name": name,
                "stage": config["experiment"]["stage"],
                "description": spec.get("description", ""),
                "base": param_base,
                "baseline_value": baseline_value,
                "values": values,
            }
        )
    return runs


def run_sensitivity(config_path: Path) -> Path:
    config = load_yaml(config_path)
    exp = config["experiment"]
    data_root = Path(exp.get("data_root", DEFAULT_DATA_ROOT)).expanduser()
    tarball = Path(exp.get("tarball", DEFAULT_TARBALL)).expanduser()
    output_dir = Path(exp["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = config_path.resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = tuple(exp.get("labels", ["yes", "no"]))
    ensure_dataset(data_root, tarball, labels)
    files = select_audio_files(
        data_root=data_root,
        labels=labels,
        files_per_label=exp.get("files_per_label"),
        max_files=exp.get("max_files"),
        selection=exp.get("selection", "first"),
        seed=int(exp.get("seed", 0)),
    )

    base_cfg = apply_config_updates(ExperimentConfig(), config.get("base_config", {}))
    parameter_runs = build_parameter_runs(config, base_cfg)
    rng = np.random.default_rng(int(exp.get("seed", 0)))

    model_cache: dict[str, Any] = {}
    transcript_cache: dict[tuple[str, str], str] = {}
    audio_cache = {path: load_audio(path, base_cfg.sample_rate) for path in files}
    rows: list[dict[str, Any]] = []

    manifest = {"config_path": str(config_path), "config": config, "files": [str(p) for p in files]}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for run in parameter_runs:
        name = run["name"]
        print(f"\n=== {run['stage']} / {name} ===")
        for value in run["values"]:
            cfg = with_parameter(run["base"], name, value)
            if cfg.whisper_model not in model_cache:
                print(f"Loading Whisper model: {cfg.whisper_model}")
                model_cache[cfg.whisper_model] = whisper.load_model(cfg.whisper_model)
            model = model_cache[cfg.whisper_model]

            for wav_path in files:
                label = wav_path.parent.name
                audio = audio_cache[wav_path]
                input_audio = add_noise(audio, cfg, rng)
                rep, rep_meta, source_mel = encode_representation(input_audio, cfg)
                recon_mel = decode_to_mel(rep, rep_meta, cfg)
                recon_audio = reconstruct_audio(recon_mel, cfg, len(input_audio))

                original_cache_key = (cfg.whisper_model, str(wav_path))
                if original_cache_key not in transcript_cache:
                    transcript_cache[original_cache_key] = transcribe_audio(model, audio)
                original_transcript = transcript_cache[original_cache_key]
                recon_transcript = transcribe_audio(model, recon_audio)

                stem = f"{run['stage']}_{name}_{value_key(value)}_{label}_{wav_path.stem}"
                recon_path = output_dir / "reconstructed" / f"{stem}.wav"
                if exp.get("save_recon_audio", False):
                    recon_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(recon_path, recon_audio, cfg.sample_rate, subtype="PCM_16")

                frames = min(source_mel.shape[1], recon_mel.shape[1])
                mel_mse = float(np.mean((source_mel[:, :frames] - recon_mel[:, :frames]) ** 2))
                original_wer = word_error_rate(label, original_transcript)
                recon_wer = word_error_rate(label, recon_transcript)

                row = {
                    "stage": run["stage"],
                    "parameter": name,
                    "parameter_value": value_key(value),
                    "baseline_value": value_key(run["baseline_value"]),
                    "file": str(wav_path),
                    "label": label,
                    "duration_s": len(audio) / cfg.sample_rate,
                    **asdict(cfg),
                    "win_length": cfg.win_length,
                    "hop_length": cfg.hop_length,
                    "n_fft": cfg.n_fft,
                    "representation_shape": "x".join(map(str, rep.shape)),
                    "representation_values": int(rep.size),
                    "compression_rate": compression_rate(audio, rep, cfg),
                    "waveform_snr_db": waveform_snr_db(input_audio, recon_audio),
                    "mel_mse": mel_mse,
                    "original_transcript": original_transcript,
                    "original_wer": original_wer,
                    "original_command_accuracy": command_accuracy(label, original_transcript),
                    "reconstructed_transcript": recon_transcript,
                    "reconstructed_wer": recon_wer,
                    "reconstructed_command_accuracy": command_accuracy(label, recon_transcript),
                    "asr_accuracy": max(0.0, 1.0 - recon_wer),
                    "reconstructed_wav": str(recon_path) if exp.get("save_recon_audio", False) else "",
                }
                rows.append(row)
            print(f"{name}={value_key(value)} done")

    raw_path = output_dir / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize_results(raw_path, output_dir, config)
    print(f"\nWrote raw results to {raw_path}")
    return raw_path


def summarize_results(raw_path: Path, output_dir: Path, config: dict[str, Any]) -> None:
    df = pd.read_csv(raw_path)
    group_cols = ["stage", "parameter", "parameter_value", "baseline_value"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            num_files=("file", "count"),
            average_accuracy=("reconstructed_command_accuracy", "mean"),
            average_asr_accuracy=("asr_accuracy", "mean"),
            average_wer=("reconstructed_wer", "mean"),
            average_original_accuracy=("original_command_accuracy", "mean"),
            average_compression_rate=("compression_rate", "mean"),
            average_waveform_snr_db=("waveform_snr_db", "mean"),
            average_mel_mse=("mel_mse", "mean"),
        )
        .reset_index()
    )
    summary_path = output_dir / "summary_by_value.csv"
    summary.to_csv(summary_path, index=False)

    efficiency = add_change_efficiency(summary, config)
    efficiency_path = output_dir / "change_efficiency.csv"
    efficiency.to_csv(efficiency_path, index=False)

    write_recommendations(efficiency, output_dir, config)
    write_plots(efficiency, output_dir)


def add_change_efficiency(summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for parameter, sub in summary.groupby("parameter", dropna=False):
        baseline_value = str(sub["baseline_value"].iloc[0])
        base_rows = sub[sub["parameter_value"].astype(str) == baseline_value]
        if base_rows.empty:
            base = sub.iloc[0]
        else:
            base = base_rows.iloc[0]
        for _, row in sub.iterrows():
            accuracy_drop = float(base["average_accuracy"] - row["average_accuracy"])
            wer_increase = float(row["average_wer"] - base["average_wer"])
            compression_gain = float(row["average_compression_rate"] - base["average_compression_rate"])
            if accuracy_drop <= 0 and compression_gain > 0:
                efficiency = float("inf")
            else:
                efficiency = compression_gain / max(accuracy_drop, 1e-6)
            out = row.to_dict()
            out.update(
                {
                    "baseline_accuracy": float(base["average_accuracy"]),
                    "baseline_wer": float(base["average_wer"]),
                    "baseline_compression_rate": float(base["average_compression_rate"]),
                    "accuracy_drop": accuracy_drop,
                    "wer_increase": wer_increase,
                    "compression_gain": compression_gain,
                    "change_efficiency": efficiency,
                    "region_label": classify_region(row, accuracy_drop, compression_gain, config),
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def classify_region(row: pd.Series, accuracy_drop: float, compression_gain: float, config: dict[str, Any]) -> str:
    analysis = config.get("analysis", {})
    safe_drop = float(analysis.get("safe_accuracy_drop", 0.02))
    useful_drop = float(analysis.get("useful_accuracy_drop", 0.05))
    collapse_drop = float(analysis.get("collapse_accuracy_drop", 0.15))
    min_gain = float(analysis.get("min_useful_compression_gain", 0.10))
    if accuracy_drop >= collapse_drop:
        return "collapse"
    if accuracy_drop <= safe_drop and compression_gain >= min_gain:
        return "safe_compression_gain"
    if accuracy_drop <= safe_drop:
        return "safe_but_little_compression_gain"
    if accuracy_drop <= useful_drop and compression_gain >= min_gain:
        return "useful_tradeoff"
    return "danger_or_low_value"


def numeric_or_none(value: Any) -> float | None:
    try:
        if str(value).lower() == "null":
            return None
        return float(value)
    except ValueError:
        return None


def write_recommendations(eff: pd.DataFrame, output_dir: Path, config: dict[str, Any]) -> None:
    lines = ["# Change-Efficiency Recommendations", ""]
    for parameter, sub in eff.groupby("parameter", dropna=False):
        sub = sub.copy()
        sub["numeric_value"] = sub["parameter_value"].map(numeric_or_none)
        good = sub[sub["region_label"].isin(["safe_compression_gain", "useful_tradeoff", "safe_but_little_compression_gain"])]
        collapse = sub[sub["region_label"] == "collapse"]
        lines += [f"## {parameter}", ""]
        if not good.empty and good["numeric_value"].notna().all():
            lo, hi = good["numeric_value"].min(), good["numeric_value"].max()
            lines.append(f"- Suggested worth-testing range: `{lo:g}` to `{hi:g}`.")
        elif not good.empty:
            values = ", ".join(f"`{v}`" for v in good["parameter_value"].tolist())
            lines.append(f"- Suggested worth-testing values: {values}.")
        else:
            lines.append("- No clearly safe/useful region found in this sweep.")

        numeric = sub.dropna(subset=["numeric_value"]).sort_values("numeric_value")
        if len(numeric) >= 3:
            gaps = np.diff(numeric["numeric_value"].to_numpy())
            median_gap = float(np.median(gaps))
            lines.append(f"- Default next-step increment: around `{median_gap:g}` in the stable region.")
            jumps = numeric[numeric["accuracy_drop"].diff().abs() >= 0.05]
            if not jumps.empty:
                focus = ", ".join(f"`{v:g}`" for v in jumps["numeric_value"].tolist())
                lines.append(f"- Use smaller increments around transition values: {focus}.")
        else:
            lines.append("- Increment suggestion: categorical parameter, keep tested categories or add specific new model/method choices.")

        if not collapse.empty:
            values = ", ".join(f"`{v}`" for v in collapse["parameter_value"].tolist())
            lines.append(f"- Collapse/extreme region observed at: {values}.")
        best = sub.sort_values(["region_label", "change_efficiency"], ascending=[True, False]).head(1).iloc[0]
        lines.append(
            f"- Best observed compression/accuracy candidate: `{best['parameter_value']}` "
            f"(accuracy={best['average_accuracy']:.3f}, compression={best['average_compression_rate']:.3f})."
        )
        lines.append("")
    (output_dir / "recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def write_plots(eff: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for parameter, sub in eff.groupby("parameter", dropna=False):
        sub = sub.copy()
        sub["numeric_value"] = sub["parameter_value"].map(numeric_or_none)
        is_numeric = sub["numeric_value"].notna().all()
        if is_numeric:
            sub = sub.sort_values("numeric_value")
            x = sub["numeric_value"]
        else:
            sub = sub.sort_values("parameter_value")
            x = np.arange(len(sub))

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(x, sub["average_accuracy"], marker="o", label="Average command accuracy", color="#1f77b4")
        ax1.set_ylabel("Average accuracy")
        ax1.set_ylim(-0.05, 1.05)
        ax2 = ax1.twinx()
        ax2.plot(x, sub["average_compression_rate"], marker="s", label="Compression rate", color="#d62728")
        ax2.set_ylabel("Compression rate")
        if not is_numeric:
            ax1.set_xticks(x)
            ax1.set_xticklabels(sub["parameter_value"], rotation=30, ha="right")
        ax1.set_xlabel(parameter)
        ax1.set_title(f"{parameter}: accuracy vs compression")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_filename(parameter)}_accuracy_compression.png", dpi=160)
        plt.close(fig)


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


if __name__ == "__main__":
    run_sensitivity(Path(parse_args().config))
