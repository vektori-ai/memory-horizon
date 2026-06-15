"""LongMemEval → memory_horizon Trajectory converter.

LongMemEval (xiaowu0162/longmemeval-cleaned) has 500 QA instances.
Each instance: a question + a haystack of sessions the model must have
processed to answer correctly.

Split:
    Train : first 60 QA instances (supplement to LoCoMo train)
    Val   : next 40 instances
    Test  : remaining 400 instances (mismatch eval — model never trained on this dist)

Question types:
    single-session-user       → easy
    single-session-assistant  → easy
    single-session-preference → easy
    knowledge-update          → medium  (fact changed across sessions)
    temporal-reasoning        → medium
    multi-session             → hard

Usage:
    python longmemeval_converter.py
    python longmemeval_converter.py --src data/longmemeval_s.json --out data/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mh_types import QAPair, Session, Trajectory, Turn

N_TRAIN = 60
N_VAL   = 40
RANDOM_SEED = 42

QTYPE_TO_DIFFICULTY = {
    "single-session-user":       "easy",
    "single-session-assistant":  "easy",
    "single-session-preference": "easy",
    "knowledge-update":          "medium",
    "temporal-reasoning":        "medium",
    "multi-session":             "hard",
}


def convert(src_path: str | Path, out_dir: str | Path) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src_path) as f:
        raw = json.load(f)

    # Stratified split — sample proportionally from each question type
    # so train/val/test all have representative coverage of memory skills
    rng = random.Random(RANDOM_SEED)
    by_type: dict[str, list] = {}
    for item in raw:
        by_type.setdefault(item["question_type"], []).append(item)

    train_raw, val_raw, test_raw = [], [], []
    for qtype, items in by_type.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * N_TRAIN / len(raw)))
        n_val   = max(1, round(n * N_VAL   / len(raw)))
        train_raw.extend(shuffled[:n_train])
        val_raw.extend(shuffled[n_train:n_train + n_val])
        test_raw.extend(shuffled[n_train + n_val:])

    counts = {}
    for split, items in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        trajs = [_item_to_trajectory(item, idx) for idx, item in enumerate(items)]
        path  = out_dir / f"longmemeval_{split}.jsonl"
        with open(path, "w") as f:
            for t in trajs:
                f.write(json.dumps(t.to_dict()) + "\n")
        counts[split] = len(trajs)
        print(f"  {split}: {len(trajs)} trajectories → {path}")

    return counts


def _item_to_trajectory(item: dict, idx: int) -> Trajectory:
    qid          = item["question_id"]
    sessions_raw = item["haystack_sessions"]    # list of sessions, each a list of turns
    dates        = item.get("haystack_dates", [])
    session_ids  = item.get("haystack_session_ids", [])
    answer_sids  = item.get("answer_session_ids", [])

    sessions = []
    for i, raw_turns in enumerate(sessions_raw):
        date = dates[i] if i < len(dates) else None
        sid  = session_ids[i] if i < len(session_ids) else f"session_{i}"

        turns = []
        for t in raw_turns:
            role    = t.get("role", "user")
            content = t.get("content", "").strip()
            if not content:
                continue
            turns.append(Turn(role=role, content=content, timestamp=date))

        if turns:
            sessions.append(Session(
                session_id=sid,
                turns=turns,
                metadata={"date": date},
            ))

    difficulty = QTYPE_TO_DIFFICULTY.get(item.get("question_type", ""), "easy")
    requires   = [
        sid for sid in answer_sids
        if any(s.session_id == sid for s in sessions)
    ]

    qa_probes = [QAPair(
        question=item["question"],
        answer=item["answer"],
        difficulty=difficulty,
        requires_sessions=requires,
        answer_type="free_form",
    )]

    return Trajectory(
        trajectory_id=f"lme_{qid}",
        sessions=sessions,
        qa_probes=qa_probes,
        vertical="longmemeval",
        metadata={
            "question_id": qid,
            "question_type": item.get("question_type", ""),
            "source": "longmemeval_s",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/longmemeval_s.json")
    parser.add_argument("--out", default="data/")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    src  = root / args.src
    out  = root / args.out

    print(f"Converting {src} → {out}")
    counts = convert(src, out)
    print(f"\nDone. Train {counts['train']} / Val {counts['val']} / Test {counts['test']}")
