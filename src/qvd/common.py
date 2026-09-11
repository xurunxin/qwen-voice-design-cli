import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Failure(Exception):
    def __init__(self, code, message, exit_code=1, details=None):
        super().__init__(message)
        self.code, self.exit_code, self.details = code, exit_code, details


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class InstanceLock:
    """Kernel-owned lock: automatically released after a process crash."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        try:
            self.file.seek(0)
            if self.file.read(1) == b"":
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise Failure("SERVICE_LOCKED", "该数据目录已有服务运行")
        return self

    def __exit__(self, *args):
        self.file.close()
