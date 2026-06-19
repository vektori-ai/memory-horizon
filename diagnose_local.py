"""Local (no GPU, no Modal) reward-coverage diagnostic.

Ported from train_modal.py::diagnose — that function's body never used any
Modal-specific calls, only the @app.local_entrypoint() decorator was
Modal-specific. Kept as a standalone script so this check survives the AWS
migration (train_modal.py is deleted once the AWS path is proven).

Usage:
    python3 diagnose_local.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent_loop import build_verl_batch
from memory_fs import VirtualFilesystem, score_trajectory

DATA_PATH_LOCAL = ROOT / "data" / "train.jsonl"


def _stats(vals: list[float]) -> str:
    n = len(vals)
    if not n:
        return "no data"
    return f"mean={sum(vals)/n:.3f}  min={min(vals):.3f}  max={max(vals):.3f}  >0: {sum(1 for v in vals if v>0)/n:.1%}"


def main() -> None:
    if not DATA_PATH_LOCAL.exists():
        print(f"No data at {DATA_PATH_LOCAL}")
        print("Run: python3 data/download_data.py")
        return

    trajectories = [
        json.loads(line) for line in DATA_PATH_LOCAL.read_text().strip().split("\n") if line.strip()
    ]
    print(f"\n=== Dataset: {len(trajectories)} trajectories ===")

    rows = build_verl_batch(trajectories)
    print(f"Training rows after window filter: {len(rows)}")

    # Reward floor: empty FS (model ABSTAINs everything)
    # Reward ceiling: model stores the answer verbatim into the FS
    floors, ceilings = [], []
    for row in rows[:200]:
        probes = row["extra_info"]["qa_probes"]
        if not probes:
            continue

        fs_empty = VirtualFilesystem()
        floors.append(score_trajectory(fs_empty, probes))

        fs_oracle = VirtualFilesystem()
        for p in probes:
            fs_oracle.apply_op(
                {"op": "STORE_FACT", "path": "facts/oracle", "content": str(p.get("answer", ""))},
                session_idx=0, turn_idx=0,
            )
        ceilings.append(score_trajectory(fs_oracle, probes))

    print(f"\nReward floor  (empty FS):  {_stats(floors)}")
    print(f"Reward ceiling (oracle FS): {_stats(ceilings)}")

    probe_counts = [len(r["extra_info"]["qa_probes"]) for r in rows]
    if probe_counts:
        print(f"\nProbes per window: mean={sum(probe_counts)/len(probe_counts):.1f}  "
              f"max={max(probe_counts)}  1-probe: {sum(1 for c in probe_counts if c==1)/len(probe_counts):.1%}")

    verticals = Counter(r["data_source"] for r in rows)
    print(f"\nVertical split: {dict(verticals)}")


if __name__ == "__main__":
    main()
