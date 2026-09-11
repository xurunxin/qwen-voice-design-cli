import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(["uv", "build", "--out-dir", str(ROOT / "dist")], cwd=ROOT, check=True)
with zipfile.ZipFile(ROOT / "dist" / "qwen-voice-design-cli-source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ("src", "scripts", "tests", "docs"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(ROOT))
    for name in ("pyproject.toml", "README.md", "models.lock.json", "Dockerfile", "compose.yaml", ".gitignore", ".dockerignore"):
        z.write(ROOT / name, name)
