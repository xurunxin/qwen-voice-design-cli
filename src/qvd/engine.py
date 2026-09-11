import json
import math
import random
import struct
import wave
from pathlib import Path

from .common import Failure


class Cancelled(Exception):
    """Raised when a queued generation was cancelled at a safe checkpoint."""


def model_revision(model_dir, configured=None):
    """Resolve a pinned local model marker when the CLI omitted revision."""
    if configured:
        return configured
    path = Path(str(model_dir)) / "qvd-model-lock.json"
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        revision = value.get("revision")
        return str(revision) if revision else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _seed_everything(seed: int, torch=None):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _model_type(model):
    return getattr(getattr(model, "model", None), "tts_model_type", None)


def _as_waveform(value):
    """Convert the official single waveform returned by qwen-tts."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        import numpy as np

        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError("VoiceDesign 返回的音频不是单声道 waveform")
        if not np.isfinite(value).all() or value.size == 0:
            raise ValueError("模型未生成有效音频")
        return value
    except ImportError:
        return value


class QwenEngine:
    """Thin adapter around qwen-tts' VoiceDesign model."""

    def __init__(self, config):
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:
            raise Failure("MODEL_LOAD_FAILED", f"无法加载 Qwen3-TTS 依赖: {exc}") from exc

        dtype_name = config.get("dtype", "bfloat16")
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise Failure("INVALID_MODEL_CONFIG", f"不支持的 dtype: {dtype_name}") from exc
        model_dir = config.get("model_dir") or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        revision = model_revision(model_dir, config.get("revision"))
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_dir,
                revision=revision,
                device_map=config.get("device", "cuda"),
                dtype=dtype,
                attn_implementation=config.get("attention", "sdpa"),
            )
        except Exception as exc:
            raise Failure("MODEL_LOAD_FAILED", f"无法加载 VoiceDesign 模型: {exc}") from exc

        model_type = _model_type(self.model)
        if model_type != "voice_design":
            raise Failure("MODEL_TYPE_INVALID", "加载的模型不是 VoiceDesign 模型")
        self.model_id = str(getattr(self.model, "name_or_path", model_dir))
        self.revision = revision
        self.dtype = dtype_name
        self.sample_rate = None

    def generate(self, request, output, progress):
        import soundfile as sf
        import torch

        _seed_everything(request["seed"], torch)
        progress(0.05, "generating")
        result = self.model.generate_voice_design(
            text=request["text"],
            language=request["language"],
            instruct=request["instruct"],
            do_sample=True,
            temperature=request["temperature"],
            top_p=request["top_p"],
            top_k=request["top_k"],
            max_new_tokens=request["max_new_tokens"],
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("VoiceDesign 返回格式无效，预期为 (wavs, sample_rate)")
        wavs, sample_rate = result
        if not isinstance(wavs, (list, tuple)) or len(wavs) != 1:
            raise ValueError("VoiceDesign 单任务应返回一个 waveform")
        if not isinstance(sample_rate, (int, float)) or int(sample_rate) <= 0:
            raise ValueError("VoiceDesign 返回了无效采样率")
        sample_rate = int(sample_rate)
        waveform = wavs[0]
        waveform = _as_waveform(waveform)
        sf.write(str(output), waveform, sample_rate, format="WAV", subtype="PCM_16")
        self.sample_rate = sample_rate
        progress(0.95, "audio written")
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "sample_rate": sample_rate,
        }


class TestEngine:
    """Explicit deterministic transport fixture; never a Qwen fallback."""

    model_id = "test-fixture"
    revision = None
    dtype = "float32"

    def __init__(self, config):
        self.config = config

    def generate(self, request, output, progress):
        sample_rate = 22_050
        duration = 0.15
        progress(0.25, "test tone (not speech)")
        frames = int(sample_rate * duration)
        with wave.open(str(Path(output)), "wb") as wav:
            wav.setparams((1, 2, sample_rate, frames, "NONE", "not compressed"))
            wav.writeframes(
                b"".join(
                    struct.pack("<h", int(2800 * math.sin(i * 440 * 2 * math.pi / sample_rate)))
                    for i in range(frames)
                )
            )
        progress(0.95, "audio written")
        return {"model_id": self.model_id, "revision": self.revision, "dtype": self.dtype, "sample_rate": sample_rate}
