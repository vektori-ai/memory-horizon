"""Convert sanitized call JSONs → memory_horizon Trajectory JSONL.

Each call becomes one Trajectory with:
  - One session (the call transcript)
  - QA probes derived from disposition keyPoints + structured fields
  - Conflict pairs where signal shifts within the call (e.g. NO_COMMITMENT → PTP)
  - Oracle memory from disposition + callBackTime + payment fields

Edge cases handled:
  - BLANK_CALL / WRONG_NUMBER / LANGUAGE_BARRIER → abstain-only probes (no storable facts)
  - Missing diarizedTranscript → skip
  - JSON parse errors → skip with warning

Usage::

    python -m memory_horizon.generator.call_converter \\
        --input  data/calls-sanitized/ \\
        --output data/debt_collection.jsonl

    # Or from Python:
    from memory_horizon.generator.call_converter import convert_directory
    convert_directory("data/calls-sanitized/", "data/debt_collection.jsonl")
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any

from memory_horizon.mh_types import (
    ConflictExample,
    MemoryOp,
    QAPair,
    Session,
    Trajectory,
    Turn,
)

# Dispositions where there are no storable facts — only abstain probes are valid.
_ABSTAIN_DISPOSITIONS = {"BLANK_CALL", "WRONG_NUMBER", "LANGUAGE_BARRIER"}

# Map from disposition value → conflict op when signal upgrades within a call.
# A keyPoint that suggests one thing early and another later = UPDATE or SUPERSEDE.
_SIGNAL_TO_OP: dict[str, MemoryOp] = {
    ("NO_COMMITMENT", "PTP"):          MemoryOp.UPDATE,
    ("NO_COMMITMENT", "STRONGEST_PTP"): MemoryOp.UPDATE,
    ("INQUIRY", "PTP"):                MemoryOp.UPDATE,
    ("INQUIRY", "STRONGEST_PTP"):      MemoryOp.UPDATE,
    ("DISPUTE", "PTP"):                MemoryOp.SUPERSEDE,
    ("DISPUTE", "STRONGEST_PTP"):      MemoryOp.SUPERSEDE,
    ("PTP", "STRONGEST_PTP"):          MemoryOp.UPDATE,
    ("DISPUTE", "ALREADY_PAID"):       MemoryOp.SUPERSEDE,
}


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_call(raw: dict[str, Any]) -> Trajectory | None:
    """Convert one call dict → Trajectory. Returns None if call should be skipped."""
    transcript = raw.get("diarizedTranscript", [])
    if not transcript:
        return None

    disp = raw.get("disposition", {})
    disp_value = disp.get("value", "UNKNOWN") if isinstance(disp, dict) else "UNKNOWN"
    call_id = raw.get("_id", str(uuid.uuid4()))
    is_abstain_call = disp_value in _ABSTAIN_DISPOSITIONS

    # Build session turns from diarized transcript.
    turns = _build_turns(transcript)
    if not turns:
        return None

    session = Session(
        session_id=f"call_{call_id[:8]}",
        turns=turns,
        metadata={
            "vertical": "debt_collection",
            "disposition": disp_value,
            "duration": raw.get("duration", 0),
            "source_id": call_id,
        },
    )

    # Build oracle memory (ground truth facts that should be stored).
    oracle_memory = _build_oracle_memory(raw, disp_value, disp)

    # Build QA probes.
    qa_probes = _build_qa_probes(raw, disp_value, disp, is_abstain_call)

    # Build conflict examples from signal shifts in keyPoints.
    conflict_examples = [] if is_abstain_call else _build_conflict_examples(disp)

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        sessions=[session],
        qa_probes=qa_probes,
        oracle_memory=oracle_memory,
        vertical="debt_collection",
        conflict_examples=conflict_examples,
        metadata={
            "source_file": call_id,
            "disposition": disp_value,
            "is_abstain_call": is_abstain_call,
            "n_turns": len(turns),
        },
    )


def convert_directory(input_dir: str | Path, output_path: str | Path) -> int:
    """Convert all call JSONs in input_dir → JSONL at output_path.

    Returns number of trajectories written.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with output_path.open("w") as f:
        for call_file in sorted(input_dir.glob("call_*.json")):
            try:
                raw = json.loads(call_file.read_text())
            except json.JSONDecodeError as e:
                print(f"[SKIP] {call_file.name}: JSON parse error — {e}")
                skipped += 1
                continue

            traj = convert_call(raw)
            if traj is None:
                print(f"[SKIP] {call_file.name}: no usable transcript")
                skipped += 1
                continue

            f.write(json.dumps(traj.to_dict()) + "\n")
            written += 1
            print(f"[OK]   {call_file.name} → {traj.metadata['disposition']} "
                  f"({len(traj.sessions[0].turns)} turns, "
                  f"{len(traj.qa_probes)} probes, "
                  f"{len(traj.conflict_examples)} conflicts)")

    print(f"\nDone: {written} trajectories written, {skipped} skipped → {output_path}")
    return written


# ---------------------------------------------------------------------------
# Turn builder
# ---------------------------------------------------------------------------

def _build_turns(transcript: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    for seg in transcript:
        text = seg.get("text", "").strip()
        if not text or text in {"****", "***", ""}:
            continue
        speaker = seg.get("speaker", "")
        role = "assistant" if speaker == "Agent" else "user"
        timestamp = None
        if "start" in seg:
            timestamp = f"t={seg['start']:.1f}s"
        turns.append(Turn(role=role, content=text, timestamp=timestamp))
    return turns


# ---------------------------------------------------------------------------
# Oracle memory builder
# ---------------------------------------------------------------------------

def _build_oracle_memory(
    raw: dict, disp_value: str, disp: dict
) -> dict[str, Any]:
    if disp_value in _ABSTAIN_DISPOSITIONS:
        return {"call_outcome": disp_value}

    oracle: dict[str, Any] = {"call_outcome": disp_value}

    # Payment amount.
    amount = disp.get("paymentAmount")
    if amount:
        oracle["payment_amount_rupees"] = amount

    # Payment date.
    pay_dt = disp.get("paymentDateAndTime")
    if pay_dt:
        oracle["payment_date"] = pay_dt

    # Callback time.
    cb = raw.get("callBackTime", {})
    if isinstance(cb, dict) and cb.get("value"):
        oracle["callback_date"] = cb["value"]
        if cb.get("remarks"):
            oracle["callback_remarks"] = cb["remarks"]

    # Broken PTP flag.
    if disp.get("isBrokenPtp"):
        oracle["is_broken_ptp"] = True

    # Grievance.
    if disp.get("isGrievance"):
        oracle["grievance_category"] = disp.get("grievanceCategory", "unspecified")

    # Summary.
    summary = raw.get("summary", "")
    if summary and isinstance(summary, str):
        oracle["call_summary"] = summary.strip()

    # Evidence quotes (what customer actually said).
    evidence = disp.get("evidence", [])
    if evidence:
        oracle["key_customer_quotes"] = evidence

    return oracle


# ---------------------------------------------------------------------------
# QA probe builder
# ---------------------------------------------------------------------------

def _build_qa_probes(
    raw: dict, disp_value: str, disp: dict, is_abstain_call: bool
) -> list[QAPair]:
    probes: list[QAPair] = []

    if is_abstain_call:
        # For no-fact calls: model should ABSTAIN on all memory queries.
        probes.append(QAPair(
            question="What payment amount did the customer commit to?",
            answer="ABSTAIN",
            difficulty="easy",
            answer_type="free_form",
        ))
        probes.append(QAPair(
            question="Did the customer make a promise to pay?",
            answer="No",
            difficulty="easy",
            answer_type="boolean",
        ))
        return probes

    # --- Outcome probe (always present) ---
    probes.append(QAPair(
        question="What was the outcome of this call?",
        answer=disp_value,
        difficulty="easy",
        answer_type="free_form",
        requires_sessions=[f"call_{raw.get('_id', '')[:8]}"],
    ))

    # --- Payment amount ---
    amount = disp.get("paymentAmount")
    if amount:
        probes.append(QAPair(
            question="What payment amount was discussed or agreed upon?",
            answer=f"{amount} rupees",
            difficulty="easy",
            answer_type="free_form",
        ))

    # --- Payment date ---
    pay_dt = disp.get("paymentDateAndTime")
    if pay_dt:
        probes.append(QAPair(
            question="What specific payment date did the customer commit to?",
            answer=pay_dt,
            difficulty="medium",
            answer_type="free_form",
        ))

    # --- Callback date ---
    cb = raw.get("callBackTime", {})
    if isinstance(cb, dict) and cb.get("value"):
        probes.append(QAPair(
            question="When should the next follow-up call be made?",
            answer=cb["value"],
            difficulty="medium",
            answer_type="free_form",
        ))

    # --- Grievance ---
    if disp.get("isGrievance"):
        probes.append(QAPair(
            question="Did the customer raise a grievance or dispute?",
            answer="Yes",
            difficulty="easy",
            answer_type="boolean",
        ))
        cat = disp.get("grievanceCategory")
        if cat:
            probes.append(QAPair(
                question="What type of grievance did the customer raise?",
                answer=cat,
                difficulty="medium",
                answer_type="free_form",
            ))

    # --- Probes from keyPoints (medium/hard) ---
    key_points = disp.get("keyPoints", [])
    for kp in key_points[:3]:  # cap at 3 to avoid redundancy
        point = kp.get("point", "")
        if not point or len(point) < 15:
            continue
        suggests = kp.get("suggests", "")
        if suggests in _ABSTAIN_DISPOSITIONS:
            continue
        probes.append(QAPair(
            question=f"Is the following true about this call: '{point}'?",
            answer="Yes",
            difficulty="hard",
            answer_type="boolean",
        ))

    # --- Summary probe (hard — requires synthesis) ---
    summary = raw.get("summary", "")
    if summary and isinstance(summary, str) and len(summary) > 30:
        probes.append(QAPair(
            question="Summarize what happened in this call regarding the customer's payment.",
            answer=summary.strip(),
            difficulty="hard",
            answer_type="free_form",
        ))

    return probes


# ---------------------------------------------------------------------------
# Conflict example builder
# ---------------------------------------------------------------------------

def _build_conflict_examples(disp: dict) -> list[ConflictExample]:
    key_points = disp.get("keyPoints", [])
    if len(key_points) < 2:
        return []

    conflicts: list[ConflictExample] = []
    signals = [kp.get("suggests", "") for kp in key_points]

    # Walk keyPoints in order — look for signal shifts that imply a memory update.
    for i in range(len(key_points) - 1):
        early_signal = signals[i]
        late_signal = signals[i + 1]
        op = _SIGNAL_TO_OP.get((early_signal, late_signal))
        if op is None:
            continue

        early_point = key_points[i].get("point", "")
        late_point = key_points[i + 1].get("point", "")
        if not early_point or not late_point:
            continue

        conflict_type = _op_to_conflict_type(op)

        conflicts.append(ConflictExample(
            old_memory={"key": "customer_payment_intent", "content": early_point},
            new_claim=late_point,
            correct_op=op,
            why=(
                f"Signal shifted from {early_signal} to {late_signal} within the call. "
                f"The customer's position changed — {_op_reason(op, early_signal, late_signal)}"
            ),
            conflict_type=conflict_type,
        ))

    return conflicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op_to_conflict_type(op: MemoryOp) -> str:
    if op == MemoryOp.UPDATE:
        return "value_updated"
    if op == MemoryOp.SUPERSEDE:
        return "factual_correction"
    if op == MemoryOp.DECAY:
        return "temporal_shift"
    return "complementary_view"


def _op_reason(op: MemoryOp, early: str, late: str) -> str:
    if op == MemoryOp.UPDATE:
        return f"old ({early}) was accurate at the time but is now superseded by new state ({late})."
    if op == MemoryOp.SUPERSEDE:
        return f"old ({early}) was effectively wrong given the new information ({late})."
    return f"transition from {early} to {late}."


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert call JSONs to memory_horizon JSONL")
    parser.add_argument("--input",  required=True, help="Directory of call_*.json files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()
    convert_directory(args.input, args.output)
