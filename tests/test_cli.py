import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from qvd.cli import parser


def invoke(home, *args):
    result = subprocess.run([sys.executable, "-m", "qvd.cli", "--home", str(home), *map(str, args)],
                            capture_output=True, text=True, encoding="utf-8", timeout=40, check=False,
                            env={**os.environ, "PYTHONUTF8": "1"})
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.stderr
    return result.returncode, lines


@pytest.fixture
def local_service(tmp_path, monkeypatch):
    monkeypatch.delenv("QVD_URL", raising=False)
    monkeypatch.delenv("QVD_API_KEY", raising=False)
    home = tmp_path / "server"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code, events = invoke(home, "service", "start", "--engine", "test", "--runtime-python", sys.executable,
                          "--port", port, "--wait", 20)
    assert code == 0, events
    try:
        yield home, events[-1]["url"]
    finally:
        code, events = invoke(home, "service", "stop", "--wait", 20)
        assert code == 0, events


def test_real_tcp_client_separate_home_design_resume_export(local_service, tmp_path, monkeypatch):
    server_home, url = local_service
    home = tmp_path / "client"
    # The CLI operates without access to server paths in requests.
    monkeypatch.setenv("QVD_API_KEY", (server_home / "local-key").read_text())
    code, _ = invoke(home, "config", "set", "--url", url)
    assert code == 0
    output = tmp_path / "audition"
    code, events = invoke(home, "design", "--text", "你好，欢迎试听。", "--instruct", "温暖的成年女性声音。",
                          "--candidates", 2, "--output-dir", output, "--wait", 20)
    assert code == 0, events
    first = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(first["candidates"]) == 2
    assert [c["request"]["seed"] for c in first["candidates"]] == [42, 43]
    for c in first["candidates"]:
        assert hashlib.sha256(Path(c["output"]).read_bytes()).hexdigest() == c["result"]["sha256"]
        assert c["result"]["backend"] == "test"
    code, events = invoke(home, "design", "--output-dir", output, "--resume", "--wait", 20)
    assert code == 0, events
    second = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [c["job_id"] for c in first["candidates"]] == [c["job_id"] for c in second["candidates"]]
    jid = first["candidates"][0]["job_id"]
    code, events = invoke(home, "voices", "save", "warm", "--job", jid)
    assert code == 0, events
    exported = tmp_path / "export"
    code, events = invoke(home, "voices", "export", "warm", "--output-dir", exported)
    assert code == 0, events
    assert (exported / "reference.txt").read_text(encoding="utf-8") == "你好，欢迎试听。"
    assert (exported / "reference.wav").read_bytes() == (output / "candidate-01.wav").read_bytes()
    code, events = invoke(home, "jobs", "submit", "--voice", "warm", "--text", "新的试听。", "--wait", 20)
    assert code == 0, events
    assert events[-1]["job"]["request"]["instruct"] == "温暖的成年女性声音。"
    assert events[-1]["job"]["request"]["text"] == "新的试听。"
    code, events = invoke(home, "jobs", "download", jid, "--output", exported / "reference.wav")
    assert code == 2 and events[-1]["error"]["code"] == "OUTPUT_EXISTS"


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_invalid_wait(value):
    from qvd.common import Failure
    with pytest.raises(Failure):
        parser().parse_args(["jobs", "watch", "abc", "--wait", value])


def test_json_errors_no_model_dependency(tmp_path):
    code, events = invoke(tmp_path, "unknown-command")
    assert code == 2
    assert events[-1]["error"]["code"] == "INVALID_ARGUMENT"
