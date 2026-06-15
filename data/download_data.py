"""Download LoCoMo + LongMemEval from HuggingFace and run converters.

Usage:
    python data/download_data.py

Writes:
    data/locomo10.json          (raw LoCoMo, 10 conversations)
    data/longmemeval_s.json     (raw LongMemEval, 500 QA instances)
    data/locomo_train.jsonl     (Trajectory format)
    data/locomo_test.jsonl
    data/longmemeval_train.jsonl
    data/longmemeval_test.jsonl
    data/train.jsonl            (merged train: locomo + longmemeval)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

def download_locomo() -> Path:
    out = DATA / "locomo10.json"
    if out.exists():
        print(f"[locomo] already at {out}, skipping download")
        return out

    print("[locomo] downloading snap-research/locomo from HuggingFace ...")
    from datasets import load_dataset
    ds = load_dataset("snap-research/locomo", split="test", trust_remote_code=True)

    conversations = []
    for row in ds:
        conversations.append({
            "sample_id": row["sample_id"],
            "conversation": row["conversation"],
            "qa": row.get("qa", []),
        })

    out.write_text(json.dumps(conversations, ensure_ascii=False, indent=2))
    print(f"[locomo] saved {len(conversations)} conversations → {out}")
    return out


def convert_locomo(src: Path) -> None:
    from converters.locomo_converter import convert
    print("[locomo] converting ...")
    counts = convert(src, DATA)
    print(f"[locomo] done — train {counts['train']} / val {counts['val']} / test {counts['test']} QA pairs")


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

def download_longmemeval() -> Path:
    out = DATA / "longmemeval_s.json"
    if out.exists():
        print(f"[longmemeval] already at {out}, skipping download")
        return out

    print("[longmemeval] downloading xiaowu0162/longmemeval from HuggingFace ...")
    from datasets import load_dataset

    # Try the -s (single-needle, 500 QA) split first; fall back to full dataset
    try:
        ds = load_dataset("xiaowu0162/longmemeval", "longmemeval_s", split="test", trust_remote_code=True)
    except Exception:
        ds = load_dataset("xiaowu0162/longmemeval", split="test", trust_remote_code=True)

    rows = [dict(row) for row in ds]
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"[longmemeval] saved {len(rows)} instances → {out}")
    return out


def convert_longmemeval(src: Path) -> None:
    from converters.longmemeval_converter import convert
    print("[longmemeval] converting ...")
    counts = convert(src, DATA)
    print(f"[longmemeval] done — train {counts['train']} / val {counts['val']} / test {counts['test']} trajectories")


# ---------------------------------------------------------------------------
# Merge train splits
# ---------------------------------------------------------------------------

def merge_train() -> None:
    out = DATA / "train.jsonl"
    sources = [
        DATA / "locomo_train.jsonl",
        DATA / "longmemeval_train.jsonl",
    ]
    lines = []
    for src in sources:
        if not src.exists():
            print(f"[merge] missing {src} — skipping")
            continue
        with open(src) as f:
            chunk = [l for l in f.read().strip().split("\n") if l.strip()]
        lines.extend(chunk)
        print(f"[merge] {src.name}: {len(chunk)} trajectories")

    out.write_text("\n".join(lines) + "\n")
    print(f"[merge] train.jsonl: {len(lines)} total trajectories → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    locomo_src = download_locomo()
    convert_locomo(locomo_src)

    lme_src = download_longmemeval()
    convert_longmemeval(lme_src)

    merge_train()
    print("\nDone. Run: modal run train_modal.py::sanity")
