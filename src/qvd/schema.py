from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LANGUAGES = (
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
    "Auto",
)


class JobRequest(BaseModel):
    """The stable request passed to the Qwen VoiceDesign generator."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    text: str = Field(min_length=1, max_length=2000)
    instruct: str = Field(min_length=1, max_length=2000)
    language: Literal["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian",
                      "Portuguese", "Spanish", "Italian", "Auto"] = "Chinese"
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    temperature: float = Field(default=0.9, gt=0, le=2.0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    max_new_tokens: int = Field(default=2048, ge=128, le=4096)

    @model_validator(mode="after")
    def validate_text(self):
        if not self.text.strip():
            raise ValueError("text 不能仅含空白")
        if not self.instruct.strip():
            raise ValueError("instruct 不能仅含空白")
        return self


class VoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    job_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_name(self):
        if not self.name.strip() or any(ch in self.name for ch in ("/", "\\", "\x00")):
            raise ValueError("音色名不能为空，且不能包含路径分隔符")
        return self


def capabilities(engine: str = "qwen", config: dict | None = None):
    config = config or {}
    model_id = config.get("model_id") or config.get("model_dir") or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    return {
        "service": "qwen-voice-design",
        "engine": engine,
        "backend": engine,
        "model": model_id,
        "model_id": model_id,
        "model_type": "VoiceDesign",
        "revision": config.get("revision"),
        "languages": list(LANGUAGES),
        "audio": {"format": "WAV", "subtype": "PCM_16", "sample_rate": None, "channels": 1},
        "format": "PCM WAV",
        "concurrency": 1,
        "voices": {"kind": "recipe_snapshot", "speaker_identity_guarantee": False},
        "progress": "phase progress; not a wall-clock ETA",
        "request_schema": JobRequest.model_json_schema(),
    }
