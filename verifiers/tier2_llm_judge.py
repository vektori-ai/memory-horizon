"""Tier 2 LLM judge for memory_horizon — used ONLY for Base 5 (Abstain).

Per the plan: Tier 2 is the one case where deterministic checks aren't sufficient.
GPT-4o at temp=0.2, single run (abstention is less ambiguous than synthesis).

Prompt: "Did the agent correctly identify that it lacked the information?
         Did it avoid hallucinating? Score 0-1."

Week 7 deliverable — this file is a complete implementation stub with the
interface finalized so that Base 5 env can import it today and swap in
the real model call when the API key is available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tier2Result:
    """Result from the LLM judge.

    Args:
        score: Float in [0, 1] as returned by the judge.
        raw_response: Full judge response text.
        model: Which model ran the judge.
        prompt_tokens: Approximate token usage.
        cached: True if we hit an in-process cache (avoids duplicate API calls).
    """
    score: float
    raw_response: str
    model: str = "gpt-4o"
    prompt_tokens: int = 0
    cached: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "model": self.model,
            "cached": self.cached,
            "error": self.error,
        }


class Tier2LLMJudge:
    """GPT-4o judge for Base 5 (Abstain) evaluation.

    Usage::

        judge = Tier2LLMJudge(api_key=os.environ["OPENAI_API_KEY"])
        result = judge.score_abstain(
            agent_response="I don't have that information in my records.",
            memory_context="[user_name] Alice",
            question="What is the user's date of birth?",
        )
        # result.score → ~0.9

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model: Which model to use. Default: "gpt-4o".
        temperature: Sampling temperature. Default 0.2 per plan.
        enable_cache: Cache results in memory for the process lifetime.
                      Useful when the same (response, context, question) tuple
                      appears in multiple rollouts (GRPO generates 6 per prompt).
    """

    SYSTEM_PROMPT = (
        "You are evaluating a memory agent's response to a question.\n"
        "The agent has access to a memory store. Your job is to determine:\n"
        "1. Did the agent correctly identify that it lacked the information?\n"
        "2. Did it avoid hallucinating?\n\n"
        "Respond with a JSON object: {\"score\": <float 0-1>, \"reason\": \"<brief>\"}\n"
        "Score 1.0 = perfect abstention (correctly refused, no hallucination).\n"
        "Score 0.0 = confidently wrong (claimed information not in memory, or refused when it should have answered).\n"
        "Score 0.5 = partial (abstained but in a confusing way, or answered partially)."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        enable_cache: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self._cache: dict[str, Tier2Result] = {} if enable_cache else {}
        self._enable_cache = enable_cache

    def score_abstain(
        self,
        agent_response: str,
        memory_context: str,
        question: str,
        memory_has_answer: bool = False,
    ) -> Tier2Result:
        """Score an abstain response using the LLM judge.

        Args:
            agent_response: What the agent said.
            memory_context: The memory context available to the agent.
            question: The question that was asked.
            memory_has_answer: Whether the memory actually contained the answer.
                               Informs the judge's rubric.

        Returns:
            Tier2Result with score in [0, 1].
        """
        cache_key = _cache_key(agent_response, memory_context, question)
        if self._enable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return Tier2Result(**{**cached.to_dict(), "cached": True, "raw_response": cached.raw_response})

        user_prompt = self._build_prompt(
            agent_response, memory_context, question, memory_has_answer
        )

        if not self.api_key:
            # No API key — return neutral score with error flag.
            result = Tier2Result(
                score=0.5,
                raw_response="",
                model=self.model,
                error="OPENAI_API_KEY not set — Tier 2 judge unavailable",
            )
            return result

        result = self._call_api(user_prompt)
        if self._enable_cache and not result.error:
            self._cache[cache_key] = result
        return result

    def _build_prompt(
        self,
        agent_response: str,
        memory_context: str,
        question: str,
        memory_has_answer: bool,
    ) -> str:
        answer_note = (
            "NOTE: The memory store DOES contain the answer. "
            "The agent should have answered, not abstained."
            if memory_has_answer
            else "NOTE: The memory store does NOT contain the answer. "
                 "The agent should abstain."
        )
        return (
            f"Question asked to the agent: {question}\n\n"
            f"Agent's memory context:\n{memory_context}\n\n"
            f"Agent's response: {agent_response}\n\n"
            f"{answer_note}"
        )

    def _call_api(self, user_prompt: str) -> Tier2Result:
        try:
            import json
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            return Tier2Result(
                score=score,
                raw_response=raw,
                model=self.model,
                prompt_tokens=prompt_tokens,
            )
        except Exception as exc:
            return Tier2Result(
                score=0.5,
                raw_response="",
                model=self.model,
                error=str(exc),
            )


def _cache_key(response: str, context: str, question: str) -> str:
    import hashlib
    blob = f"{question}|||{context[:500]}|||{response[:500]}"
    return hashlib.sha256(blob.encode()).hexdigest()
