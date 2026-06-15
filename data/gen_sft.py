"""SFT demonstration generator using GPT-OSS 120B (OpenAI API).

Processes LoCoMo train conversations turn by turn, asks GPT-OSS to emit
the correct memory op JSON for each turn. Saves valid traces as SFT examples.

Memory-R1 used 152 QA pairs for SFT and hit strong results — we target 200
traces here. Each trace = one turn + correct op. Cheap: ~$0.15 per 200 turns
at gpt-oss-120b pricing.

Usage:
    export OPENAI_API_KEY=sk-...
    python data/gen_sft.py                          # generates data/sft_traces.jsonl
    python data/gen_sft.py --n 300 --out data/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
DEFAULT_OUT = DATA / "sft_traces.jsonl"

MODEL = "gpt-oss-120b"   # OpenAI GPT-OSS 120B

# ── System prompt for the strong model ──────────────────────────────────────
# Same schema our Qwen3-8B will be trained on.

_SFT_SYSTEM = """\
You are an expert memory manager for a long-running conversational AI.
Given the current memory filesystem state and a new conversation turn,
output the single best memory operation as a JSON object.

Memory filesystem paths:
  people/   — facts about individuals
  events/   — past and future events
  places/   — locations and addresses
  facts/    — general facts and context
  prefs/    — preferences and habits

Output exactly one JSON object, nothing else:

Write ops:
  {"op": "STORE_FACT",  "path": "category/entity", "content": "concise fact to store"}
  {"op": "UPDATE",      "path": "category/entity", "content": "updated value"}
  {"op": "SUPERSEDE",   "path": "category/entity", "content": "new value that replaces all prior"}
  {"op": "COMPRESS",    "path": "category/entity", "content": "summary replacing verbose entries"}
  {"op": "ABSTAIN"}

Retrieval op (when you need to check memory before deciding):
  {"op": "RETRIEVE", "query": "what to look up"}

Rules:
- Use SUPERSEDE when a fact directly contradicts something previously stored.
- Use UPDATE when a fact refines or extends something stored.
- Use ABSTAIN when the turn contains no storable information (greetings, filler, etc).
- Keep content concise — one sentence max.
- path must be category/entity (e.g. people/alice, facts/payment_plan).
"""


def _make_user_prompt(fs_state: str, turn_text: str) -> str:
    return f"[Memory state]\n{fs_state}\n\n[Current turn]\n{turn_text}"


def _is_valid_op(obj: dict) -> bool:
    """Check the op is schema-valid for our training format."""
    op = obj.get("op", "")
    valid_ops = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS", "ABSTAIN", "RETRIEVE"}
    if op not in valid_ops:
        return False
    if op in ("STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS"):
        if not obj.get("content") or not obj.get("path"):
            return False
        if "/" not in obj.get("path", ""):
            return False
    if op == "RETRIEVE":
        if not obj.get("query"):
            return False
    return True


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ── Simple VirtualFilesystem for SFT generation (no verl dependency) ────────

class _FS:
    def __init__(self):
        self.files: dict[str, list[str]] = {}

    def apply(self, op: dict) -> None:
        t = op.get("op", "")
        path = op.get("path", "").strip()
        content = op.get("content", "").strip()
        if t in ("UPDATE", "SUPERSEDE", "COMPRESS") and path and content:
            self.files[path] = [content]
        elif t == "STORE_FACT" and path and content:
            self.files.setdefault(path, []).append(content)

    def render(self) -> str:
        if not self.files:
            return "(memory is empty)"
        lines = []
        for path in sorted(self.files):
            lines.append(f"=== {path} ===")
            lines.extend(self.files[path])
        return "\n".join(lines)


# ── Generation ───────────────────────────────────────────────────────────────

def generate(
    locomo_path: Path,
    out_path: Path,
    n_target: int = 200,
    max_per_conv: int = 40,
    retry_invalid: int = 2,
) -> int:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(locomo_path) as f:
        raw = json.load(f)

    # Use only the train conversation (conv-26) — same as Memory-R1
    train_convs = [c for c in raw if c.get("sample_id") in {"conv-26"}]
    if not train_convs:
        # fall back to first conversation if split not found
        train_convs = raw[:1]
        print(f"[warn] conv-26 not found, using first conversation")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces_written = 0

    with open(out_path, "w") as out_f:
        for conv in train_convs:
            if traces_written >= n_target:
                break

            cdata   = conv["conversation"]
            spk_a   = cdata.get("speaker_a", "PersonA")
            session_nums = sorted(
                int(k.split("_")[1])
                for k in cdata
                if re.match(r"^session_\d+$", k)
            )

            fs = _FS()
            conv_traces = 0

            for sn in session_nums:
                if traces_written >= n_target or conv_traces >= max_per_conv:
                    break
                for turn in cdata.get(f"session_{sn}", []):
                    if traces_written >= n_target or conv_traces >= max_per_conv:
                        break

                    speaker = turn.get("speaker", "")
                    role    = "User" if speaker == spk_a else "Agent"
                    text    = turn.get("text", "").strip()
                    if not text:
                        continue

                    turn_str  = f"{role}: {text}"
                    fs_render = fs.render()
                    user_msg  = _make_user_prompt(fs_render, turn_str)

                    op_dict = None
                    for attempt in range(1 + retry_invalid):
                        try:
                            resp = client.chat.completions.create(
                                model=MODEL,
                                messages=[
                                    {"role": "system",  "content": _SFT_SYSTEM},
                                    {"role": "user",    "content": user_msg},
                                ],
                                max_tokens=128,
                                temperature=0.0,
                            )
                            raw_text = resp.choices[0].message.content or ""
                            parsed   = _parse_json(raw_text)
                            if parsed and _is_valid_op(parsed):
                                op_dict = parsed
                                break
                            if attempt < retry_invalid:
                                time.sleep(0.5)
                        except Exception as e:
                            print(f"  [api error] {e}")
                            time.sleep(2)

                    if op_dict is None:
                        continue  # skip invalid — don't poison SFT data

                    # SFT example: prompt → completion
                    trace = {
                        "messages": [
                            {"role": "system",    "content": _SFT_SYSTEM},
                            {"role": "user",      "content": user_msg},
                            {"role": "assistant", "content": json.dumps(op_dict)},
                        ],
                        "op":     op_dict.get("op"),
                        "source": f"locomo_{conv.get('sample_id')}",
                    }
                    out_f.write(json.dumps(trace) + "\n")
                    out_f.flush()

                    fs.apply(op_dict)
                    traces_written += 1
                    conv_traces    += 1

                    if traces_written % 25 == 0:
                        print(f"  [{traces_written}/{n_target}] traces written")

    print(f"\nDone. {traces_written} SFT traces → {out_path}")
    return traces_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/locomo10.json",  help="raw LoCoMo JSON")
    parser.add_argument("--out", default=str(DEFAULT_OUT),      help="output JSONL")
    parser.add_argument("--n",   type=int, default=200,         help="target trace count")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY first")
        sys.exit(1)

    src = ROOT / args.src
    if not src.exists():
        print(f"Error: {src} not found — run python data/download_data.py first")
        sys.exit(1)

    count = generate(src, Path(args.out), n_target=args.n)
    print(f"SFT traces: {count}. Next: modal run train_modal.py::sft")
