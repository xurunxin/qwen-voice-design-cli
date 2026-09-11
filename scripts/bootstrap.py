"""Compatibility launcher; deployment logic lives in qvd init, including uv bootstrap."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qvd.provision import Installer

p = argparse.ArgumentParser(description="安装 CLI 并调用 qvd init；无 Python 的 Windows 请用 install.ps1")
p.add_argument("--home", type=Path, default=Path.home() / ".qwen-voice-design")
p.add_argument("--cpu", action="store_true")
p.add_argument("--skip-cli", action="store_true", help="通过 uv tool run 临时安装启动器，不注册全局 CLI")
args, remaining = p.parse_known_args()
installer = Installer(args.home, "qvd", lambda event: print(json.dumps(event), flush=True))
uv = installer.ensure_uv()
if args.skip_cli:
    command = [uv, "tool", "run", "--from", str(ROOT), "qvd"]
else:
    installer.run("cli", [uv, "tool", "install", "--force", "--editable", ROOT])
    directory = subprocess.check_output([uv, "tool", "dir", "--bin"], text=True).strip()
    command = [str(Path(directory) / ("qvd.exe" if os.name == "nt" else "qvd"))]
command.extend(["--home", str(args.home), "init"])
if args.cpu:
    command.extend(["--device", "cpu"])
raise SystemExit(subprocess.run([*command, *remaining], check=False).returncode)
