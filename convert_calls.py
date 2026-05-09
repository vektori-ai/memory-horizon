"""Standalone script: convert sanitized call JSONs → debt_collection.jsonl

Run from anywhere:
    python3 convert_calls.py

No package imports — avoids the types.py / stdlib shadowing issue.
Outputs data/debt_collection.jsonl in the memory_horizon Trajectory format.
"""

import json
import re
import uuid
from pathlib import Path

REAL_CALLS_DIR = Path(__file__).parent / "data" / "calls-sanitized (1)" / "calls-sanitized"
SYNTH_CALLS_DIR = Path(__file__).parent / "data" / "synthetic_calls"
OUTPUT_PATH = Path(__file__).parent / "data" / "debt_collection.jsonl"

ABSTAIN_DISPOSITIONS = {"BLANK_CALL", "WRONG_NUMBER", "LANGUAGE_BARRIER"}

SIGNAL_TO_OP = {
    ("NO_COMMITMENT", "PTP"):           "UPDATE",
    ("NO_COMMITMENT", "STRONGEST_PTP"): "UPDATE",
    ("INQUIRY",       "PTP"):           "UPDATE",
    ("INQUIRY",       "STRONGEST_PTP"): "UPDATE",
    ("DISPUTE",       "PTP"):           "SUPERSEDE",
    ("DISPUTE",       "STRONGEST_PTP"): "SUPERSEDE",
    ("PTP",           "STRONGEST_PTP"): "UPDATE",
    ("DISPUTE",       "ALREADY_PAID"):  "SUPERSEDE",
}

OP_TO_CONFLICT_TYPE = {
    "UPDATE":    "value_updated",
    "SUPERSEDE": "factual_correction",
    "DECAY":     "temporal_shift",
    "KEEP_BOTH": "complementary_view",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_call(raw: dict) -> dict | None:
    transcript = raw.get("diarizedTranscript", [])
    if not transcript:
        return None

    disp = raw.get("disposition", {}) or {}
    disp_value = disp.get("value", "UNKNOWN")
    call_id = raw.get("_id", str(uuid.uuid4()))
    is_abstain = disp_value in ABSTAIN_DISPOSITIONS

    turns = build_turns(transcript)
    if not turns:
        return None

    session = {
        "session_id": f"call_{call_id[:8]}",
        "turns": turns,
        "memory_snapshot": None,
        "metadata": {
            "vertical": "debt_collection",
            "disposition": disp_value,
            "duration": raw.get("duration", 0),
            "source_id": call_id,
        },
    }

    oracle_memory = build_oracle_memory(raw, disp_value, disp)
    qa_probes = build_qa_probes(raw, disp_value, disp, is_abstain, call_id)
    conflict_examples = [] if is_abstain else build_conflict_examples(disp)

    return {
        "trajectory_id": str(uuid.uuid4()),
        "sessions": [session],
        "qa_probes": qa_probes,
        "oracle_memory": oracle_memory,
        "vertical": "debt_collection",
        "conflict_examples": conflict_examples,
        "metadata": {
            "source_file": call_id,
            "disposition": disp_value,
            "is_abstain_call": is_abstain,
            "n_turns": len(turns),
        },
    }


def build_turns(transcript: list[dict]) -> list[dict]:
    turns = []
    for seg in transcript:
        text = (seg.get("text") or "").strip()
        if not text or set(text) <= {"*", " "}:
            continue
        role = "assistant" if seg.get("speaker") == "Agent" else "user"
        ts = f"t={seg['start']:.1f}s" if "start" in seg else None
        turns.append({"role": role, "content": text, "timestamp": ts, "metadata": {}})
    return turns


def build_oracle_memory(raw: dict, disp_value: str, disp: dict) -> dict:
    if disp_value in ABSTAIN_DISPOSITIONS:
        return {"call_outcome": disp_value}

    oracle = {"call_outcome": disp_value}

    amount = disp.get("paymentAmount")
    if amount:
        oracle["payment_amount_rupees"] = amount

    pay_dt = disp.get("paymentDateAndTime")
    if pay_dt:
        oracle["payment_date"] = pay_dt

    cb = raw.get("callBackTime") or {}
    if cb.get("value"):
        oracle["callback_date"] = cb["value"]
    if cb.get("remarks"):
        oracle["callback_remarks"] = cb["remarks"]

    if disp.get("isBrokenPtp"):
        oracle["is_broken_ptp"] = True

    if disp.get("isGrievance"):
        oracle["grievance_category"] = disp.get("grievanceCategory", "unspecified")

    summary = raw.get("summary") or ""
    if isinstance(summary, str) and summary.strip():
        oracle["call_summary"] = summary.strip()

    evidence = disp.get("evidence") or []
    if evidence:
        oracle["key_customer_quotes"] = evidence

    return oracle


def build_qa_probes(raw: dict, disp_value: str, disp: dict, is_abstain: bool, call_id: str) -> list[dict]:
    probes = []
    sid = f"call_{call_id[:8]}"

    if is_abstain:
        probes.append(qa("What payment amount did the customer commit to?", "ABSTAIN", "easy"))
        probes.append(qa("Did the customer make a promise to pay?", "No", "easy", atype="boolean"))
        return probes

    probes.append(qa("What was the outcome of this call?", disp_value, "easy", requires=[sid]))

    amount = disp.get("paymentAmount")
    if amount:
        probes.append(qa("What payment amount was discussed or agreed upon?", f"{amount} rupees", "easy"))

    pay_dt = disp.get("paymentDateAndTime")
    if pay_dt:
        probes.append(qa("What specific payment date did the customer commit to?", pay_dt, "medium"))

    cb = raw.get("callBackTime") or {}
    if cb.get("value"):
        probes.append(qa("When should the next follow-up call be made?", cb["value"], "medium"))

    if disp.get("isGrievance"):
        probes.append(qa("Did the customer raise a grievance or dispute?", "Yes", "easy", atype="boolean"))
        cat = disp.get("grievanceCategory")
        if cat:
            probes.append(qa("What type of grievance did the customer raise?", cat, "medium"))

    for kp in (disp.get("keyPoints") or [])[:3]:
        point = kp.get("point", "")
        if len(point) < 15 or kp.get("suggests") in ABSTAIN_DISPOSITIONS:
            continue
        probes.append(qa(
            f"Is the following true about this call: '{point}'?",
            "Yes", "hard", atype="boolean",
        ))

    summary = raw.get("summary") or ""
    if isinstance(summary, str) and len(summary) > 30:
        probes.append(qa(
            "Summarize what happened in this call regarding the customer's payment.",
            summary.strip(), "hard",
        ))

    return probes


def qa(question: str, answer: str, difficulty: str,
       requires: list[str] | None = None, atype: str = "free_form") -> dict:
    return {
        "question": question,
        "answer": answer,
        "difficulty": difficulty,
        "requires_sessions": requires or [],
        "answer_type": atype,
        "choices": [],
        "answer_idx": None,
    }


def build_conflict_examples(disp: dict) -> list[dict]:
    key_points = disp.get("keyPoints") or []
    if len(key_points) < 2:
        return []

    conflicts = []
    signals = [kp.get("suggests", "") for kp in key_points]

    for i in range(len(key_points) - 1):
        op = SIGNAL_TO_OP.get((signals[i], signals[i + 1]))
        if not op:
            continue
        early = key_points[i].get("point", "")
        late  = key_points[i + 1].get("point", "")
        if not early or not late:
            continue
        conflicts.append({
            "old_memory": {"key": "customer_payment_intent", "content": early},
            "new_claim": late,
            "correct_op": op,
            "why": (
                f"Signal shifted from {signals[i]} → {signals[i+1]} within call. "
                f"Customer's position changed."
            ),
            "conflict_type": OP_TO_CONFLICT_TYPE.get(op, "value_updated"),
        })

    return conflicts


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    all_files = sorted(REAL_CALLS_DIR.glob("call_*.json")) + sorted(SYNTH_CALLS_DIR.glob("call_*.json"))

    with OUTPUT_PATH.open("w") as f:
        for call_file in all_files:
            try:
                raw = json.loads(call_file.read_text())
            except json.JSONDecodeError as e:
                print(f"[SKIP] {call_file.name}: JSON error — {e}")
                skipped += 1
                continue

            traj = convert_call(raw)
            if traj is None:
                print(f"[SKIP] {call_file.name}: no transcript")
                skipped += 1
                continue

            f.write(json.dumps(traj) + "\n")
            written += 1
            disp = traj["metadata"]["disposition"]
            n_turns  = traj["metadata"]["n_turns"]
            n_probes = len(traj["qa_probes"])
            n_conf   = len(traj["conflict_examples"])
            print(f"[OK]   {call_file.name}  disp={disp}  turns={n_turns}  probes={n_probes}  conflicts={n_conf}")

    print(f"\n{written} trajectories → {OUTPUT_PATH}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
