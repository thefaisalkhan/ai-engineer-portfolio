"""
Multi-Provider LLM Client with Cost + Latency Tracking
Wraps OpenAI and Anthropic APIs behind a unified interface.
Tracks: tokens, cost ($), latency (ms), errors.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ─── Pricing (as of 2024, per 1M tokens) ────────────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def cost_usd(self) -> float:
        p = PRICING.get(self.model, {"input": 0, "output": 0})
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000


@dataclass
class UsageStats:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    errors: int = 0
    responses: list[LLMResponse] = field(default_factory=list)

    def record(self, response: LLMResponse):
        self.total_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.total_latency_ms += response.latency_ms
        self.responses.append(response)

    def summary(self) -> str:
        avg_latency = self.total_latency_ms / max(self.total_calls, 1)
        return (
            f"Calls: {self.total_calls} | "
            f"Tokens: {self.total_input_tokens}in / {self.total_output_tokens}out | "
            f"Cost: ${self.total_cost_usd:.4f} | "
            f"Avg latency: {avg_latency:.0f}ms | "
            f"Errors: {self.errors}"
        )


# ─── Base Client ─────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.stats = UsageStats()

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        raise NotImplementedError

    def _record(self, response: LLMResponse):
        self.stats.record(response)
        return response


# ─── OpenAI Client ───────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__(model)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def complete(self, prompt: str, system: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 1024) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return self._mock_response(prompt)

        client = AsyncOpenAI(api_key=self._api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = await client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        response = LLMResponse(
            content=resp.choices[0].message.content,
            model=self.model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return self._record(response)

    def _mock_response(self, prompt: str) -> LLMResponse:
        response = LLMResponse(
            content=f"[Mock OpenAI] Response to: {prompt[:50]}...",
            model=self.model,
            input_tokens=len(prompt.split()),
            output_tokens=20,
            latency_ms=150.0,
        )
        return self._record(response)


# ─── Anthropic Client ────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        super().__init__(model)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def complete(self, prompt: str, system: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 1024) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            return self._mock_response(prompt)

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        start = time.perf_counter()
        resp = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        response = LLMResponse(
            content=resp.content[0].text,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return self._record(response)

    def _mock_response(self, prompt: str) -> LLMResponse:
        response = LLMResponse(
            content=f"[Mock Claude] Response to: {prompt[:50]}...",
            model=self.model,
            input_tokens=len(prompt.split()),
            output_tokens=20,
            latency_ms=200.0,
        )
        return self._record(response)


# ─── Cost Comparison ─────────────────────────────────────────────────────────

async def compare_models(prompt: str) -> None:
    """Run same prompt across models, compare cost + latency."""
    clients = [
        OpenAIClient("gpt-4o-mini"),
        OpenAIClient("gpt-4o"),
        AnthropicClient("claude-3-haiku-20240307"),
        AnthropicClient("claude-3-5-sonnet-20241022"),
    ]

    print(f"\n=== Model Comparison ===")
    print(f"Prompt: '{prompt[:60]}...'\n")

    results = await asyncio.gather(*[c.complete(prompt) for c in clients], return_exceptions=True)

    print(f"{'Model':<40} {'Latency':>10} {'In tok':>8} {'Out tok':>8} {'Cost':>10}")
    print("-" * 80)
    for r in results:
        if isinstance(r, LLMResponse):
            print(
                f"{r.model:<40} {r.latency_ms:>8.0f}ms {r.input_tokens:>8} "
                f"{r.output_tokens:>8} ${r.cost_usd:>9.5f}"
            )

    # Cumulative stats
    print("\n=== Usage Stats per Client ===")
    for c in clients:
        print(f"  {c.model:<40} {c.stats.summary()}")


if __name__ == "__main__":
    asyncio.run(compare_models(
        "Explain retrieval-augmented generation in 3 sentences."
    ))
