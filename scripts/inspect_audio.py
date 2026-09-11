"""Inspect generated candidates and create a portable listening page."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
args = p.parse_args()
directory = args.directory.resolve()
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
rows = []
cards = []
for candidate in manifest["candidates"]:
    path = directory / f"candidate-{int(candidate['index']):02d}.wav"
    samples, sr = sf.read(path, always_2d=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == candidate["result"]["sha256"], "checksum mismatch"
    assert samples.size and np.isfinite(samples).all(), "invalid audio"
    row = {"file": path.name, "sample_rate": sr, "channels": samples.shape[1],
           "duration_seconds": len(samples) / sr, "peak": float(np.max(np.abs(samples))),
           "rms": float(np.sqrt(np.mean(samples**2))),
           "clipped_samples": int(np.sum(np.abs(samples) >= 32767 / 32768)),
           "sha256": digest, "backend": candidate["result"]["backend"],
           "generation_seconds": candidate["result"]["generation_seconds"]}
    rows.append(row)
    cards.append(f'<article><h2>Candidate {candidate["index"]} · seed {candidate["request"]["seed"]}</h2>'
                 f'<p>{html.escape(candidate["request"]["instruct"])}</p>'
                 f'<blockquote>{html.escape(candidate["request"]["text"])}</blockquote>'
                 f'<audio controls src="{path.name}"></audio>'
                 f'<p>{row["duration_seconds"]:.2f}s · {sr} Hz · {html.escape(row["backend"])} · '
                 f'{row["generation_seconds"]:.1f}s generation</p></article>')
report = {"audio": rows, "unique_hashes": len({r["sha256"] for r in rows}),
          "subjective_quality": "Not rated. Listen to judge voice and instruction match."}
(directory / "audio-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Qwen Voice Design — Audition</title><style>body{font:17px/1.65 system-ui;max-width:850px;'
        'margin:48px auto;padding:0 24px;background:#f4f2ee;color:#252725}article{background:white;'
        'border-radius:16px;padding:24px;margin:24px 0}h1,h2{line-height:1.3}audio{width:100%}'
        'blockquote{margin:16px 0;color:#62665f}</style><h1>Qwen Voice Design · 音色试听</h1>'
        '<p>相同描述与台词，不同随机种子。实际听音决定音色是否符合需求；技术检查不代表主观品质验收。</p>'
        + ''.join(cards))
(directory / "index.html").write_text(page, encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
