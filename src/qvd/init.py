"""Install a managed VoiceDesign runtime from a lightweight CLI installation."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import Failure, InstanceLock, atomic_json, read_json
from .provision import Installer, safe_claim

LOCK = json.loads(Path(__file__).with_name("models.lock.json").read_text(encoding="utf-8"))
FILES = ("config.json", "generation_config.json", "merges.txt", "model.safetensors", "preprocessor_config.json",
         "speech_tokenizer/config.json", "speech_tokenizer/configuration.json", "speech_tokenizer/model.safetensors",
         "speech_tokenizer/preprocessor_config.json", "tokenizer_config.json", "vocab.json")


def arguments(sub):
    p = sub.add_parser("init", help="自动安装独立 Python、推理依赖和固定版本模型")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="只读检查，不下载/写文件")
    mode.add_argument("--dry-run", action="store_true", help="只显示安装计划")
    p.add_argument("--install-dir", type=Path, help="默认 <home>/runtime；不覆盖陌生非空目录")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--skip-models", action="store_true", help="仅准备运行时，不报告模型就绪")
    p.add_argument("--start", action="store_true", help="安装并验证完成后启动服务")
    p.add_argument("--wait", type=float, default=600)


def locations(args):
    home = Path(args.home).resolve()
    previous = read_json(home / "runtime.json", {})
    root = (args.install_dir or Path(previous.get("install_dir", home / "runtime"))).resolve()
    device = args.device or previous.get("profile", "cuda")
    return home, root, device, root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"), root / "models" / "VoiceDesign"


def model_check(model):
    missing = [name for name in FILES if not (model / name).is_file() or (model / name).stat().st_size == 0]
    inventory = read_json(model / "qvd-model-ready.json", {})
    valid_inventory = (inventory.get("revision") == LOCK["revision"] and set(inventory.get("files", {})) == set(FILES)
                       and read_json(model / "qvd-model-lock.json", {}) == LOCK)
    wrong_size = [name for name, size in inventory.get("files", {}).items() if name in FILES and
                  (not (model / name).is_file() or (model / name).stat().st_size != size)]
    return {"ready": not missing and valid_inventory and not wrong_size,
            "missing": missing, "size_mismatch": wrong_size, "revision_verified": valid_inventory}


def runtime_probe(python, device):
    if not python.is_file():
        return {"ready": False, "reason": "运行时尚未安装"}
    code = ("import json,sys; import torch,torchaudio,qwen_tts,fastapi,uvicorn,soundfile; "
            "from importlib.metadata import version; print(json.dumps({'python':sys.version_info[:2],"
            "'torch':torch.__version__,'torchaudio':torchaudio.__version__,'qwen_tts':version('qwen-tts'),"
            "'cuda':torch.cuda.is_available(),'bf16':torch.cuda.is_bf16_supported()}))")
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=120, check=False,
                                env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        data = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {}
        ready = (result.returncode == 0 and data.get("python") == [3, 11] and data.get("qwen_tts") == LOCK["qwen_tts"]
                 and data.get("torch", "").split("+")[0] == LOCK["torch"]
                 and data.get("torchaudio", "").split("+")[0] == LOCK["torchaudio"]
                 and (device == "cpu" or data.get("cuda") is True))
        return {"ready": ready, "exit_code": result.returncode, **data}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {"ready": False, "reason": "运行时导入检查失败，重新执行 init 可修复受管环境"}


def inspect(args, probe=True):
    home, root, device, python, model = locations(args)
    runtime = runtime_probe(python, device) if probe else {"ready": False, "checked": False}
    models = model_check(model)
    ownership = read_json(root / ".qvd-install.json", {}) == {"tool": "qvd", "profile": device, "lock": LOCK}
    return {"ready": runtime["ready"] and models["ready"] and ownership, "runtime": runtime, "models": models, "ownership": ownership,
            "install_dir": str(root), "runtime_python": str(python), "model_dir": str(model), "profile": device,
            "uv": shutil.which("uv"), "driver_detected": shutil.which("nvidia-smi") is not None,
            "stages": ["uv", "python-3.11", "isolated-venv", "torch", "runtime-dependencies", "models", "verify", "save-runtime"],
            "model_revision": LOCK["revision"], "log": str(home / "init.log")}


def download_models(model):
    safe_claim(model, "qvd-model-lock.json", LOCK)
    if model_check(model)["ready"]:
        return
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    for attempt in range(3):
        try:
            info = HfApi().model_info(LOCK["model"], revision=LOCK["revision"], files_metadata=True, timeout=30)
            break
        except Exception as exc:
            if attempt == 2:
                raise Failure("MODEL_MANIFEST_FAILED", "模型文件清单读取失败，重试 init 可继续", 1,
                              {"error_type": type(exc).__name__}) from exc
            time.sleep(2 ** attempt)
    inventory = {entry.rfilename: entry.size for entry in info.siblings if entry.rfilename in FILES}
    for name, size in inventory.items():
        path = model / name
        if path.is_file() and path.stat().st_size != size:
            hf_hub_download(LOCK["model"], name, revision=LOCK["revision"], local_dir=model, force_download=True)
    snapshot_download(LOCK["model"], revision=LOCK["revision"], local_dir=model,
                      allow_patterns=list(FILES), max_workers=4)
    if set(inventory) != set(FILES) or any(not size or (model / name).stat().st_size != size for name, size in inventory.items()):
        raise Failure("MODEL_INCOMPLETE", "模型文件未完整下载；重新执行 init 续传")
    atomic_json(model / "qvd-model-ready.json", {"revision": LOCK["revision"], "files": inventory})


def initialize(args, emit):
    home, root, device, python, model = locations(args)
    if args.check or args.dry_run:
        return {"mode": "check" if args.check else "dry-run", **inspect(args, probe=args.check)}
    if args.start and args.skip_models:
        raise Failure("INVALID_ARGUMENT", "--start 不能与 --skip-models 同时使用", 2)
    if device == "cuda" and not shutil.which("nvidia-smi"):
        raise Failure("GPU_DRIVER_MISSING", "找不到 NVIDIA 驱动工具 nvidia-smi；先安装 NVIDIA 驱动，或使用 init --device cpu", 2)
    if device == "cuda":
        probe = subprocess.run([shutil.which("nvidia-smi"), "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20, check=False)
        if probe.returncode or not probe.stdout.strip():
            raise Failure("GPU_DRIVER_UNAVAILABLE", "NVIDIA 驱动未能识别 GPU；修复驱动或使用 --device cpu", 2)
    if device == "cuda" and sys.platform not in {"win32", "linux"}:
        raise Failure("PLATFORM_UNSUPPORTED", "此平台请使用 --device cpu", 2)
    with InstanceLock(home / "init.lock"):
        safe_claim(root, ".qvd-install.json", {"tool": "qvd", "profile": device, "lock": LOCK})
        saved = read_json(home / "runtime.json", {})
        if saved.get("runtime_python") == str(python):
            # Do not replace loaded DLLs/packages in a running managed service.
            with InstanceLock(home / "service.lock"):
                pass
        installer = Installer(home, "qvd", emit)
        uv = installer.ensure_uv()
        # uv owns the managed interpreter cache; no system Python or PATH modification.
        installer.run("python", [uv, "python", "install", "3.11"])
        if not python.exists():
            if (root / ".venv").exists():
                raise Failure("PARTIAL_VENV", "发现未完成的 .venv；请选择新 --install-dir，保留现有目录供检查", 2)
            installer.run("venv", [uv, "venv", "--python", "3.11", root / ".venv"])
        index = LOCK["cuda_index"] if device == "cuda" else "https://download.pytorch.org/whl/cpu"
        installer.run("torch", [uv, "pip", "install", "--python", python, "torch==" + LOCK["torch"],
                                "torchaudio==" + LOCK["torchaudio"], "--index-url", index])
        wheel = installer.client_wheel("qvd", "qwen-voice-design-cli")
        installer.run("dependencies", [uv, "pip", "install", "--python", python, "--reinstall-package", "qwen-voice-design-cli", str(wheel) + "[server,model]"])
        if not args.skip_models and not model_check(model)["ready"]:
            installer.run("models", [python, "-m", "qvd.init", "--download-models", model])
        report = inspect(args)
        if not report["runtime"]["ready"] or (not args.skip_models and not report["models"]["ready"]):
            raise Failure("INIT_VERIFY_FAILED", "运行时或模型验证失败；检查驱动和 init.log 后重试", 1, report)
        config = {"install_dir": str(root), "runtime_python": str(python), "model_dir": str(model), "profile": device,
                  "device": "cuda:0" if device == "cuda" else "cpu",
                  "dtype": ("bfloat16" if report["runtime"].get("bf16") else "float16") if device == "cuda" else "float32"}
        atomic_json(home / "runtime.json", config)
        atomic_json(home / "init-state.json", {"state": "complete" if report["ready"] else "runtime_only", "ready": report["ready"]})
    if args.start:
        from .cli import background_start, parser, resolve_runtime
        start = parser().parse_args(["--home", str(home), "service", "start", "--wait", str(args.wait)])
        if args.api_key_env:
            start.api_key_env = args.api_key_env
        resolve_runtime(start)
        report["service"] = background_start(start)
    return {**report, "next": "qvd service start --wait 600" if report["ready"] else "qvd init"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download-models", required=True, type=Path)
    download_models(p.parse_args().download_models.resolve())
