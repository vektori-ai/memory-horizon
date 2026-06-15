"""LoCoMo → memory_horizon Trajectory converter.

LoCoMo (snap-research/locomo) is a multi-session personal conversation dataset.
10 conversations, 19-35 sessions each, 105-260 QA pairs per conversation.

Split (matches Memory-R1 exactly for direct benchmark comparison):
    Train : conv-26 only  (same single conversation Memory-R1 trained on)
    Val   : conv-30, conv-41
    Test  : remaining 7 conversations (~1100+ QA pairs)

QA categories:
    1 = single-hop  → difficulty "easy"
    2 = multi-hop   → difficulty "medium"
    3 = temporal    → difficulty "medium"
    4 = commonsense → difficulty "hard"
    5 = adversarial → EXCLUDED (Memory-R1 excludes these too)

Usage:
    python locomo_converter.py                        # writes to data/
    python locomo_converter.py --src data/locomo10.json --out data/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mh_types import QAPair, Session, Trajectory, Turn

# Conversations used for each split — matches Memory-R1
TRAIN_IDS = {"conv-26"}
VAL_IDS   = {"conv-30", "conv-41"}
# everything else → test

CATEGORY_TO_DIFFICULTY = {1: "easy", 2: "medium", 3: "medium", 4: "hard"}
ADVERSARIAL_CATEGORY = 5


def convert(locomo_path: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Convert locomo10.json into three JSONL files.

    Returns counts: {"train": N, "val": N, "test": N}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(locomo_path) as f:
        raw = json.load(f)

    splits: dict[str, list[Trajectory]] = {"train": [], "val": [], "test": []}

    for conv in raw:
        traj = _conv_to_trajectory(conv)
        sid = conv["sample_id"]
        if sid in TRAIN_IDS:
            splits["train"].append(traj)
        elif sid in VAL_IDS:
            splits["val"].append(traj)
        else:
            splits["test"].append(traj)

    counts = {}
    for split, trajs in splits.items():
        path = out_dir / f"locomo_{split}.jsonl"
        with open(path, "w") as f:
            for t in trajs:
                f.write(json.dumps(t.to_dict()) + "\n")
        n_qa = sum(len(t.qa_probes) for t in trajs)
        counts[split] = n_qa
        print(f"  {split}: {len(trajs)} conversations, {n_qa} QA pairs → {path}")

    return counts


def _conv_to_trajectory(conv: dict) -> Trajectory:
    cdata = conv["conversation"]
    sample_id = conv["sample_id"]

    speaker_a = cdata.get("speaker_a", "PersonA")
    speaker_b = cdata.get("speaker_b", "PersonB")

    # Extract ordered sessions
    session_nums = sorted(
        int(k.split("_")[1])
        for k in cdata
        if re.match(r"^session_\d+$", k)
    )

    sessions = []
    for n in session_nums:
        session_key = f"session_{n}"
        date_key    = f"session_{n}_date_time"

        raw_turns = cdata.get(session_key, [])
        if not raw_turns:
            continue

        timestamp = cdata.get(date_key)
        turns = []
        for t in raw_turns:
            speaker = t.get("speaker", "")
            role = "user" if speaker == speaker_a else "assistant"
            turns.append(Turn(
                role=role,
                content=t.get("text", "").strip(),
                timestamp=timestamp,
                metadata={"speaker": speaker, "dia_id": t.get("dia_id", "")},
            ))

        sessions.append(Session(
            session_id=f"{sample_id}_s{n}",
            turns=turns,
            metadata={"date": timestamp, "session_num": n},
        ))

    # QA probes — exclude adversarial (category 5)
    qa_probes = []
    for qa in conv.get("qa", []):
        cat = qa.get("category", 1)
        if cat == ADVERSARIAL_CATEGORY:
            continue

        difficulty = CATEGORY_TO_DIFFICULTY.get(cat, "easy")
        evidence   = qa.get("evidence", [])

        # evidence entries like "D1:3" → session references
        requires = list({
            f"{sample_id}_s{_parse_session_num(e)}"
            for e in evidence
            if _parse_session_num(e) is not None
        })

        qa_probes.append(QAPair(
            question=qa["question"],
            answer=str(qa["answer"]),
            difficulty=difficulty,
            requires_sessions=requires,
            answer_type="free_form",
        ))

    return Trajectory(
        trajectory_id=f"locomo_{sample_id}",
        sessions=sessions,
        qa_probes=qa_probes,
        vertical="locomo",
        metadata={"sample_id": sample_id, "source": "locomo10"},
    )


def _parse_session_num(evidence_ref: str) -> int | None:
    """Parse session number from evidence ref like 'D3:7' → 3."""
    m = re.match(r"D(\d+):", evidence_ref)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/locomo10.json")
    parser.add_argument("--out", default="data/")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    src  = root / args.src
    out  = root / args.out

    print(f"Converting {src} → {out}")
    counts = convert(src, out)
    total = sum(counts.values())
    print(f"\nDone. Total QA pairs: {total}")
    print(f"  Train {counts['train']} / Val {counts['val']} / Test {counts['test']}")
