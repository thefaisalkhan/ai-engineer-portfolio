"""
LLM-Specific Monitoring: Token usage, cost tracking, prompt/response logging
Critical for production LLM apps: controls spend, enables debugging, supports compliance.
"""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Token Cost Registry ──────────────────────────────────────────────────────

COST_PER_1M_TOKENS = {
    "gpt-4o":                    {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "claude-3-5-sonnet-20241022":{"input": 3.00,  "output": 15.00},
    "claude-3-haiku-20240307":   {"input": 0.25,  "output": 1.25},
    "text-embedding-3-small":    {"input": 0.02,  "output": 0.0},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = COST_PER_1M_TOKENS.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class LLMCallRecord:
    record_id: str
    timestamp: str
    model: str
    user_id: Optional[str]
    session_id: Optional[str]
    endpoint: str
    prompt_hash: str           # for privacy: hash, don't store raw prompt
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    # Feedback
    thumbs_up: Optional[bool] = None
    feedback_text: Optional[str] = None


@dataclass
class UsageBudget:
    daily_limit_usd: float = 10.0
    per_user_daily_limit_usd: float = 1.0
    spent_today: float = 0.0
    per_user_spent: dict = field(default_factory=dict)

    def check(self, user_id: str, estimated_cost: float) -> tuple[bool, str]:
        if self.spent_today + estimated_cost > self.daily_limit_usd:
            return False, f"Daily budget exhausted (${self.spent_today:.2f}/${self.daily_limit_usd})"
        user_spent = self.per_user_spent.get(user_id, 0.0)
        if user_spent + estimated_cost > self.per_user_daily_limit_usd:
            return False, f"User budget exhausted (${user_spent:.2f}/${self.per_user_daily_limit_usd})"
        return True, "ok"

    def record_spend(self, user_id: str, cost: float):
        self.spent_today += cost
        self.per_user_spent[user_id] = self.per_user_spent.get(user_id, 0.0) + cost


# ─── Audit Log ────────────────────────────────────────────────────────────────

class AuditLog:
    """
    Append-only log of all LLM interactions.
    Used for: debugging, compliance, cost attribution, replay.
    In production: write to S3, BigQuery, or a dedicated log DB.
    """

    def __init__(self, path: Optional[Path] = None):
        self.records: list[LLMCallRecord] = []
        self.path = path

    def write(self, record: LLMCallRecord):
        self.records.append(record)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    def cost_by_model(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for r in self.records:
            totals[r.model] = totals.get(r.model, 0.0) + r.cost_usd
        return totals

    def cost_by_user(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for r in self.records:
            uid = r.user_id or "anonymous"
            totals[uid] = totals.get(uid, 0.0) + r.cost_usd
        return totals

    def latency_stats(self) -> dict[str, float]:
        latencies = sorted(r.latency_ms for r in self.records if r.success)
        if not latencies:
            return {}
        n = len(latencies)
        return {
            "p50": latencies[n // 2],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[int(n * 0.99)],
            "mean": sum(latencies) / n,
            "min": latencies[0],
            "max": latencies[-1],
        }

    def error_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if not r.success) / len(self.records)

    def feedback_stats(self) -> dict[str, float]:
        with_feedback = [r for r in self.records if r.thumbs_up is not None]
        if not with_feedback:
            return {"feedback_rate": 0.0, "positive_rate": 0.0}
        positive = sum(1 for r in with_feedback if r.thumbs_up)
        return {
            "feedback_rate": len(with_feedback) / len(self.records),
            "positive_rate": positive / len(with_feedback),
        }

    def summary(self) -> dict:
        return {
            "total_calls": len(self.records),
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 4),
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "error_rate": round(self.error_rate(), 4),
            "latency_ms": {k: round(v, 1) for k, v in self.latency_stats().items()},
            "cost_by_model": {k: round(v, 4) for k, v in self.cost_by_model().items()},
            "cost_by_user": {k: round(v, 4) for k, v in self.cost_by_user().items()},
            "feedback": self.feedback_stats(),
        }


# ─── Monitored LLM Wrapper ────────────────────────────────────────────────────

class MonitoredLLM:
    """
    Wraps any LLM client with monitoring, budget enforcement, and audit logging.
    """

    def __init__(self, model: str, budget: Optional[UsageBudget] = None):
        self.model = model
        self.budget = budget or UsageBudget()
        self.audit = AuditLog()

    def call(
        self,
        prompt: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        endpoint: str = "/chat",
    ) -> dict:
        # Estimate cost (rough: 1 token ≈ 4 chars)
        estimated_input = len(prompt) // 4
        estimated_cost = compute_cost(self.model, estimated_input, 200)

        # Budget check
        allowed, reason = self.budget.check(user_id, estimated_cost)
        if not allowed:
            return {"error": f"Budget exceeded: {reason}", "content": None}

        # Simulate LLM call
        start = time.perf_counter()
        import random
        time.sleep(random.uniform(0.05, 0.3))
        success = random.random() > 0.05
        latency_ms = (time.perf_counter() - start) * 1000

        if success:
            response_text = f"[Simulated response to: {prompt[:50]}...]"
            output_tokens = len(response_text.split())
        else:
            response_text = None
            output_tokens = 0

        actual_cost = compute_cost(self.model, estimated_input, output_tokens)
        self.budget.record_spend(user_id, actual_cost)

        record = LLMCallRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            model=self.model,
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            endpoint=endpoint,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            input_tokens=estimated_input,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(actual_cost, 6),
            success=success,
            error=None if success else "InferenceError",
        )
        self.audit.write(record)

        return {
            "content": response_text,
            "record_id": record.record_id,
            "cost_usd": record.cost_usd,
            "latency_ms": record.latency_ms,
        }

    def add_feedback(self, record_id: str, thumbs_up: bool):
        for r in self.audit.records:
            if r.record_id == record_id:
                r.thumbs_up = thumbs_up
                return


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    llm = MonitoredLLM(
        model="gpt-4o-mini",
        budget=UsageBudget(daily_limit_usd=5.0, per_user_daily_limit_usd=0.5),
    )

    users = ["alice", "bob", "charlie", "alice", "bob"]
    prompts = [
        "Summarize the RAG architecture",
        "What is an embedding?",
        "Explain gradient descent",
        "How does attention work in transformers?",
        "What is LangChain?",
    ]

    print("=== Simulating LLM Calls ===")
    record_ids = []
    for user, prompt in zip(users, prompts):
        result = llm.call(prompt, user_id=user, endpoint="/chat")
        status = "OK" if result.get("content") else f"ERROR: {result.get('error')}"
        if result.get("record_id"):
            record_ids.append(result["record_id"])
        print(f"  [{user}] {status} | cost=${result.get('cost_usd', 0):.6f} | {result.get('latency_ms', 0):.0f}ms")

    # Add some feedback
    for rid in record_ids[:3]:
        llm.add_feedback(rid, thumbs_up=random.random() > 0.3)

    print("\n=== Audit Summary ===")
    print(json.dumps(llm.audit.summary(), indent=2))
