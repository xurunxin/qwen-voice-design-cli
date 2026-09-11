import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from qvd.cli import parser, resolve_runtime
from qvd.common import Failure, atomic_json
from qvd.init import FILES, LOCK, initialize, model_check
from qvd.provision import Installer, safe_claim, safe_extract_zip


@pytest.mark.parametrize("mode", ["--check", "--dry-run"])
def test_read_only_missing_home(tmp_path, mode):
    home = tmp_path / "new-home"
    args = parser().parse_args(["--home", str(home), "init", mode])
    result = initialize(args, lambda event: None)
    assert not result["ready"]
    assert not home.exists()


def test_ownership_and_profiles(tmp_path):
    root = tmp_path / "managed"
    safe_claim(root, "owner.json", {"device": "cuda"})
    safe_claim(root, "owner.json", {"device": "cuda"})
    with pytest.raises(Failure, match="配置不同"):
        safe_claim(root, "owner.json", {"device": "cpu"})
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "mine.txt").write_text("preserve")
    with pytest.raises(Failure, match="不会接管"):
        safe_claim(other, "owner.json", {})
    assert (other / "mine.txt").read_text() == "preserve"


def test_runtime_precedence(tmp_path, monkeypatch):
    atomic_json(tmp_path / "runtime.json", {"runtime_python": "saved/python", "model_dir": "saved/models", "device": "cpu", "dtype": "float32"})
    monkeypatch.setenv("QVD_MODEL_DIR", "environment/models")
    args = parser().parse_args(["--home", str(tmp_path), "service", "start", "--device", "cuda:1"])
    resolve_runtime(args)
    assert args.runtime_python == Path("saved/python")
    assert args.model_dir == "environment/models"
    assert args.device == "cuda:1"
    assert args.dtype == "float32"


def test_model_inventory_detects_truncation(tmp_path):
    for name in FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"abc")
    assert not model_check(tmp_path)["ready"]
    atomic_json(tmp_path / "qvd-model-lock.json", LOCK)
    atomic_json(tmp_path / "qvd-model-ready.json", {"revision": LOCK["revision"], "files": dict.fromkeys(FILES, 3)})
    assert model_check(tmp_path)["ready"]
    (tmp_path / "model.safetensors").write_bytes(b"a")
    assert model_check(tmp_path)["size_mismatch"] == ["model.safetensors"]


def test_safe_archive_rejects_escape_before_writing(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("root/good.py", "ok")
        z.writestr("root/../../escape.py", "bad")
    destination = tmp_path / "source"
    with pytest.raises(Failure):
        safe_extract_zip(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.py").exists()


def test_failed_stage_logs_and_can_retry(tmp_path):
    events = []
    install = Installer(tmp_path, "qvd", events.append)
    with pytest.raises(Failure):
        install.run("fixture", [sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"])
    assert "diagnostic" in (tmp_path / "init.log").read_text()
    assert json.loads((tmp_path / "init-state.json").read_text())["state"] == "failed"
    assert "recovered" in install.run("fixture", [sys.executable, "-c", "print('recovered')"])
    assert json.loads((tmp_path / "init-state.json").read_text())["state"] == "succeeded"


def test_client_wheel_includes_init_and_locks(tmp_path):
    path = Installer(tmp_path, "qvd", lambda event: None).client_wheel("qvd", "qwen-voice-design-cli")
    with zipfile.ZipFile(path) as wheel:
        assert "qvd/init.py" in wheel.namelist()
        assert "qvd/models.lock.json" in wheel.namelist()
        assert not any(".venv" in name or "models/" in name for name in wheel.namelist())


def test_missing_uv_download_uses_verified_binary(tmp_path, monkeypatch):
    from qvd import provision

    monkeypatch.setattr(provision.shutil, "which", lambda name: None)
    monkeypatch.setattr(provision.sys, "platform", "win32")
    monkeypatch.setattr(provision.platform, "machine", lambda: "AMD64")
    metadata = {"urls": [{"filename": "uv-py3-none-win_amd64.whl", "url": "https://fixture/uv", "digests": {"sha256": "expected"}}]}
    monkeypatch.setattr(provision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(metadata).encode()))

    def download(url, target, expected_sha256):
        assert expected_sha256 == "expected"
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as wheel:
            wheel.writestr("uv.data/scripts/" + ("uv.exe" if os.name == "nt" else "uv"), b"binary")

    monkeypatch.setattr(provision, "download", download)
    monkeypatch.setattr(Installer, "run", lambda *a, **k: "uv fixture")
    assert Path(Installer(tmp_path, "qvd", lambda event: None).ensure_uv()).read_bytes() == b"binary"


def test_failed_runtime_verification_does_not_save_config(tmp_path, monkeypatch):
    from qvd import init

    args = parser().parse_args(["--home", str(tmp_path), "init", "--device", "cpu", "--skip-models"])
    monkeypatch.setattr(Installer, "ensure_uv", lambda self: "uv")
    monkeypatch.setattr(Installer, "run", lambda *a, **k: "")
    monkeypatch.setattr(Installer, "client_wheel", lambda *a, **k: tmp_path / "client.whl")
    monkeypatch.setattr(init, "runtime_probe", lambda *a: {"ready": False, "reason": "fixture failure"})
    with pytest.raises(Failure) as error:
        initialize(args, lambda event: None)
    assert error.value.code == "INIT_VERIFY_FAILED"
    assert not (tmp_path / "runtime.json").exists()


def test_skip_models_saves_runtime_but_not_ready(tmp_path, monkeypatch):
    from qvd import init

    args = parser().parse_args(["--home", str(tmp_path), "init", "--device", "cpu", "--skip-models"])
    monkeypatch.setattr(Installer, "ensure_uv", lambda self: "uv")
    monkeypatch.setattr(Installer, "run", lambda *a, **k: "")
    monkeypatch.setattr(Installer, "client_wheel", lambda *a, **k: tmp_path / "client.whl")
    monkeypatch.setattr(init, "runtime_probe", lambda *a: {"ready": True})
    result = initialize(args, lambda event: None)
    assert not result["ready"]
    config = json.loads((tmp_path / "runtime.json").read_text())
    assert config["device"] == "cpu" and config["dtype"] == "float32"
