"""Legacy model-only entry point, using the same inventory verification as qvd init."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qvd.init import download_models

download_models(ROOT / "models" / "VoiceDesign")
