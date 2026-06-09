"""Generate synthetic debt-collection call transcripts using Gemini 2.5 Flash.

Uses 2 real calls as few-shot examples. Generates 200 synthetic calls covering
all disposition types with varied borrower profiles, amounts, and dialogue patterns.

Usage:
    python3 generate_synthetic_calls.py            # generates 200 calls
    python3 generate_synthetic_calls.py --n 20     # quick test batch

Output: data/synthetic_calls/call_NNNN_<DISPOSITION>.json
Then run: python3 convert_calls.py  (already handles this dir via --input)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_env_key(name: str) -> str:
    val = os.environ.get(name)
    if val:
        return val
    env_file = Path(__file__).parent / ".env"
    for line in env_file.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"{name} not found in environment or .env")

API_KEY   = _load_env_key("GEMINI_API_KEY")
MODEL     = "gemini-2.5-flash-lite"
API_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
OUT_DIR   = Path("data/synthetic_calls")
DELAY_SEC = 1.2   # stay under rate limit

# Disposition mix — weighted toward the interesting ones for memory training
DISPOSITIONS = [
    "PTP",            # promise to pay — core positive case
    "PTP",
    "PTP",
    "STRONGEST_PTP",  # proactive full commitment
    "STRONGEST_PTP",
    "CALLBACK",       # needs follow-up — tests temporal memory
    "CALLBACK",
    "DISPUTE",        # customer disputes debt — tests SUPERSEDE
    "DISPUTE",
    "INQUIRY",        # asking about settlement — tests partial store
    "ALREADY_PAID",   # claims payment made — tests conflict resolution
    "NO_COMMITMENT",  # engaged but no commitment — tests abstain
    "LANGUAGE_BARRIER",
    "WRONG_NUMBER",
]

# Borrower profiles to vary
PROFILES = [
    {"name": "Priya Sharma",   "amount": 45000,  "settlement": 31500, "situation": "lost job recently, looking for EMI option"},
    {"name": "Rahul Verma",    "amount": 72000,  "settlement": 50400, "situation": "says already paid some amount, disputes remaining"},
    {"name": "Anjali Mehta",   "amount": 28000,  "settlement": 19600, "situation": "willing to pay but wants time till salary"},
    {"name": "Suresh Pillai",  "amount": 95000,  "settlement": 66500, "situation": "aggressive, says institution cancelled admission"},
    {"name": "Neha Gupta",     "amount": 35000,  "settlement": 24500, "situation": "polite, asks for payment link and best price"},
    {"name": "Arun Kumar",     "amount": 58000,  "settlement": 40600, "situation": "busy, asks for callback tomorrow morning"},
    {"name": "Meera Iyer",     "amount": 42000,  "settlement": 29400, "situation": "family emergency, will pay by month end"},
    {"name": "Vikram Singh",   "amount": 67000,  "settlement": 46900, "situation": "disputes amount, says 4 EMIs only"},
    {"name": "Pooja Nair",     "amount": 33000,  "settlement": 23100, "situation": "asks about settlement options, no commitment yet"},
    {"name": "Rajesh Yadav",   "amount": 81000,  "settlement": 56700, "situation": "very cooperative, commits immediately with date"},
    {"name": "Sunita Patil",   "amount": 25000,  "settlement": 17500, "situation": "says wrong number but agent keeps trying"},
    {"name": "Kiran Reddy",    "amount": 54000,  "settlement": 37800, "situation": "speaks only Telugu, language barrier"},
    {"name": "Deepak Joshi",   "amount": 39000,  "settlement": 27300, "situation": "claims scam, refuses to pay"},
    {"name": "Kavya Patel",    "amount": 63000,  "settlement": 44100, "situation": "agrees to pay in two instalments"},
    {"name": "Amit Tiwari",    "amount": 47000,  "settlement": 32900, "situation": "says will check bank account and call back"},
]

# ---------------------------------------------------------------------------
# Few-shot examples (trimmed from real calls)
# ---------------------------------------------------------------------------

FEW_SHOT_PTP = {
    "diarizedTranscript": [
        {"start": 0, "end": 10, "text": "Hello, this is Alex from DemoCompany, calling about your DemoLender loan. We reviewed your account and have a good offer to help close it. Can we talk for a moment?", "speaker": "Agent"},
        {"start": 10, "end": 11, "text": "Yes.", "speaker": "Customer"},
        {"start": 11, "end": 28, "text": "Your total outstanding is fifty thousand rupees. But we can remove all charges and close your loan at just thirty five thousand rupees. Would you like to discuss how to resolve this today?", "speaker": "Agent"},
        {"start": 28, "end": 35, "text": "How is that possible? What charges are you removing?", "speaker": "Customer"},
        {"start": 35, "end": 60, "text": "We can waive the late fees and processing charges. The core principal is what remains. Can I ask what's been holding you back from making payments?", "speaker": "Agent"},
        {"start": 60, "end": 75, "text": "I don't have the money right now. My salary got delayed.", "speaker": "Customer"},
        {"start": 75, "end": 90, "text": "I understand. When do you expect to have the funds available?", "speaker": "Agent"},
        {"start": 90, "end": 100, "text": "By month end. I should get paid by then.", "speaker": "Customer"},
        {"start": 100, "end": 115, "text": "So end of this month you can make the payment of thirty five thousand?", "speaker": "Agent"},
        {"start": 115, "end": 120, "text": "Yes. That should work.", "speaker": "Customer"},
    ],
    "disposition": {
        "value": "PTP",
        "remarks": "I don't have the money right now | Yes. That should work.",
        "evidence": ["I don't have the money right now. My salary got delayed.", "Yes. That should work."],
        "keyPoints": [
            {"point": "Customer initially says no funds due to salary delay", "suggests": "NO_COMMITMENT"},
            {"point": "Customer confirms end of month payment when asked", "suggests": "PTP"},
        ],
        "paymentAmount": 35000,
        "paymentDateAndTime": None,
        "isGrievance": False,
    },
    "callBackTime": {"value": "2026-05-31T10:00:00.000Z", "remarks": "Customer will pay by end of month"},
    "summary": "Customer agreed to pay the settlement amount of 35,000 rupees by end of month. Salary delay was the reason for non-payment.",
}

FEW_SHOT_DISPUTE = {
    "diarizedTranscript": [
        {"start": 0, "end": 10, "text": "Hello, this is Alex from DemoCompany about your DemoLender loan. Can we talk?", "speaker": "Agent"},
        {"start": 10, "end": 12, "text": "Yes.", "speaker": "Customer"},
        {"start": 12, "end": 30, "text": "You have an outstanding of sixty thousand rupees. We can close it at forty two thousand. Does that sound good?", "speaker": "Agent"},
        {"start": 30, "end": 50, "text": "I am not paying anything. The college cancelled my admission. Why should I pay for a loan for a course I never did?", "speaker": "Customer"},
        {"start": 50, "end": 65, "text": "I understand your frustration. However the loan was disbursed and the repayment obligation remains.", "speaker": "Agent"},
        {"start": 65, "end": 85, "text": "No. We already gave notice to the institution. I am not paying. You can contact my lawyer.", "speaker": "Customer"},
    ],
    "disposition": {
        "value": "DISPUTE",
        "remarks": "I am not paying anything | You can contact my lawyer.",
        "evidence": ["I am not paying anything. The college cancelled my admission.", "You can contact my lawyer."],
        "keyPoints": [
            {"point": "Customer disputes loan obligation citing cancelled admission", "suggests": "DISPUTE"},
            {"point": "Customer refuses to pay and mentions legal action", "suggests": "DISPUTE"},
        ],
        "paymentAmount": None,
        "paymentDateAndTime": None,
        "isGrievance": True,
    },
    "callBackTime": {"value": None, "remarks": None},
    "summary": "Customer disputes the loan entirely, citing cancelled college admission. Refuses to pay and mentions lawyer.",
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(disposition: str, profile: dict) -> str:
    few_shot_1 = json.dumps(FEW_SHOT_PTP, indent=2)
    few_shot_2 = json.dumps(FEW_SHOT_DISPUTE, indent=2)

    disposition_guidance = {
        "PTP":             "Customer eventually commits to paying by a specific date, prompted by agent.",
        "STRONGEST_PTP":   "Customer proactively commits with exact date AND exact amount without being prompted.",
        "CALLBACK":        "Customer cannot pay now, requests agent to call back at a specific time tomorrow or next week.",
        "DISPUTE":         "Customer refuses to pay, disputes the debt — wrong amount, cancelled service, or scam allegation.",
        "ALREADY_PAID":    "Customer claims they already made the payment and asks why they're still being called.",
        "INQUIRY":         "Customer asks about settlement options and best price but makes no commitment.",
        "NO_COMMITMENT":   "Customer engages but conversation stalls before any payment discussion completes.",
        "LANGUAGE_BARRIER":"Customer switches to a regional language (Tamil, Telugu, or Marathi) and communication breaks down.",
        "WRONG_NUMBER":    "Person who answers is not the borrower. They say wrong number and hang up.",
    }

    guidance = disposition_guidance.get(disposition, "Customer engages normally.")

    return f"""You generate realistic synthetic debt-collection call transcripts for AI training data.

The calls are from DemoCompany calling borrowers about DemoLender education loans.
The agent is always named Alex. Amounts are in Indian rupees. Keep dialogue natural and realistic.

Here are two example calls showing the exact JSON format required:

EXAMPLE 1 (PTP — promise to pay):
{few_shot_1}

EXAMPLE 2 (DISPUTE):
{few_shot_2}

Now generate a NEW call with these parameters:
- Borrower name: {profile['name']}
- Outstanding amount: {profile['amount']} rupees
- Settlement offer: {profile['settlement']} rupees
- Borrower situation: {profile['situation']}
- Target disposition: {disposition}
- Disposition guidance: {guidance}

Rules:
- diarizedTranscript must have 8-20 turns, each with start/end/text/speaker fields
- speaker is always "Agent" or "Customer"
- start/end are cumulative seconds (increment realistically)
- disposition.keyPoints must have 2-5 points, each with "point" and "suggests" fields
- "suggests" must be one of: PTP, STRONGEST_PTP, CALLBACK, DISPUTE, ALREADY_PAID, INQUIRY, NO_COMMITMENT, LANGUAGE_BARRIER, WRONG_NUMBER
- paymentAmount: integer in rupees if committed, null otherwise
- paymentDateAndTime: ISO-8601 if specific date given, null otherwise
- callBackTime.value: ISO-8601 if callback scheduled, null otherwise
- summary: 1-2 sentences summarising what happened
- isGrievance: true only if customer raised a formal complaint (DISPUTE, scam allegation)

Return ONLY valid JSON. No markdown, no explanation, no code blocks. Just the JSON object."""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, retries: int = 3) -> str | None:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n: int = 200, start: int = 1) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check model name works — quick ping
    print(f"Model: {MODEL}")
    print(f"Generating {n} synthetic calls → {OUT_DIR}\n")

    random.seed(start)
    ok = fail = 0

    for i in range(n):
        idx = start + i
        disposition = random.choice(DISPOSITIONS)
        profile     = random.choice(PROFILES)

        prompt   = build_prompt(disposition, profile)
        raw_json = call_gemini(prompt)

        if raw_json is None:
            print(f"[{idx:04d}] FAIL  {disposition}")
            fail += 1
            continue

        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"[{idx:04d}] JSON ERROR  {disposition}: {e}")
            fail += 1
            continue

        # Inject required fields the model might omit
        obj.setdefault("_id",       f"syn_{uuid.uuid4().hex[:16]}")
        obj.setdefault("channel",   "call")
        obj.setdefault("sourceId",  "synthetic")
        obj.setdefault("duration",  int(obj.get("diarizedTranscript", [{}])[-1].get("end", 0)))
        obj.setdefault("noOfTurns", len(obj.get("diarizedTranscript", [])))
        obj.setdefault("summary",   "")
        obj.setdefault("isBotDetected",    {"value": False, "remarks": ""})
        obj.setdefault("isCustomerAgitated", {"value": False, "remarks": ""})
        obj.setdefault("isCustomerComplaining", {"value": False, "remarks": ""})

        # Ensure disposition has all required subfields
        disp = obj.setdefault("disposition", {})
        disp.setdefault("value",             disposition)
        disp.setdefault("paymentAmount",     None)
        disp.setdefault("paymentDateAndTime",None)
        disp.setdefault("isGrievance",       False)
        disp.setdefault("keyPoints",         [])
        disp.setdefault("evidence",          [])
        disp.setdefault("isBrokenPtp",       False)

        obj.setdefault("callBackTime", {"value": None, "remarks": None})

        fname = OUT_DIR / f"call_{idx:04d}_{disposition}.json"
        fname.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

        n_turns  = len(obj.get("diarizedTranscript", []))
        amount   = disp.get("paymentAmount")
        print(f"[{idx:04d}] OK    {disposition:<18} turns={n_turns}  amount={amount}  → {fname.name}")

        ok += 1
        time.sleep(DELAY_SEC)

    print(f"\nDone: {ok} generated, {fail} failed")
    print(f"Next step: update convert_calls.py to also read from {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of calls to generate")
    parser.add_argument("--start", type=int, default=1, help="Starting index (to resume)")
    args = parser.parse_args()
    generate(args.n, start=args.start)
