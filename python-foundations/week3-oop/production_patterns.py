"""
Production Python Patterns: Decorators, Generators, Async, Context Managers
These patterns appear in every production AI codebase.
"""

import asyncio
import functools
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── 1. Decorators ────────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Retry decorator — essential for LLM API calls that fail intermittently."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(f"{func.__name__} failed (attempt {attempt}), retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(f"{func.__name__} failed (attempt {attempt}), retrying in {wait}s: {e}")
                    time.sleep(wait)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def timed(func):
    """Log execution time — useful for profiling LLM inference."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__name__} took {elapsed:.1f}ms")
        return result
    return wrapper


# ─── 2. Generators ────────────────────────────────────────────────────────────

def document_stream(corpus: list[str], batch_size: int = 2) -> Iterator[list[str]]:
    """
    Memory-efficient document batching for embedding large corpora.
    Never loads all docs into memory — critical for 1M+ document ingestion.
    """
    for i in range(0, len(corpus), batch_size):
        yield corpus[i : i + batch_size]


def token_budget_filter(docs: list[str], max_tokens: int = 512) -> Iterator[str]:
    """Yield only documents within token budget (rough estimate: 1 token ≈ 4 chars)."""
    for doc in docs:
        estimated_tokens = len(doc) // 4
        if estimated_tokens <= max_tokens:
            yield doc
        else:
            logger.warning(f"Skipping doc (est. {estimated_tokens} tokens > {max_tokens})")


# ─── 3. Async/Await ───────────────────────────────────────────────────────────

@dataclass
class FakeLLMClient:
    """Simulates concurrent LLM API calls."""
    model: str = "gpt-4o"
    latency_ms: float = 100.0

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(self.latency_ms / 1000)
        return f"[{self.model}] Response to: {prompt[:30]}..."


async def batch_embed_async(texts: list[str], client: FakeLLMClient) -> list[str]:
    """
    Concurrent LLM calls — 10x faster than sequential for batch processing.
    Pattern used in embedding services and evaluation pipelines.
    """
    tasks = [client.complete(text) for text in texts]
    return await asyncio.gather(*tasks)


async def stream_response(prompt: str) -> AsyncIterator[str]:
    """Simulate streaming LLM response — used in chatbot UIs."""
    tokens = f"Response to '{prompt}': Hello world this is a streamed response".split()
    for token in tokens:
        await asyncio.sleep(0.05)
        yield token + " "


# ─── 4. Context Managers ──────────────────────────────────────────────────────

@contextmanager
def llm_session(model: str):
    """
    Context manager for LLM session lifecycle.
    Ensures cleanup on exception — pattern for DB connections, file handles, API sessions.
    """
    logger.info(f"Opening LLM session: {model}")
    session = {"model": model, "tokens_used": 0, "calls": 0}
    try:
        yield session
    finally:
        logger.info(
            f"Closing session: {session['calls']} calls, "
            f"{session['tokens_used']} tokens used"
        )


# ─── 5. Dataclasses for Config ────────────────────────────────────────────────

@dataclass
class RAGConfig:
    """Typed config — avoids magic strings throughout codebase."""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5
    similarity_threshold: float = 0.75
    max_tokens: int = 2048
    temperature: float = 0.0
    tags: list[str] = field(default_factory=list)

    def is_strict_mode(self) -> bool:
        return self.temperature == 0.0 and self.similarity_threshold >= 0.75


# ─── Demo ────────────────────────────────────────────────────────────────────

@timed
def demo_generators():
    corpus = [f"Document {i} about AI engineering topics." for i in range(10)]
    filtered = list(token_budget_filter(corpus, max_tokens=100))
    batches = list(document_stream(filtered, batch_size=3))
    print(f"\n=== Generators ===")
    print(f"Corpus: {len(corpus)} docs → filtered: {len(filtered)} → batches: {len(batches)}")


async def demo_async():
    client = FakeLLMClient(latency_ms=50)
    prompts = [f"Summarize document {i}" for i in range(5)]

    start = time.perf_counter()
    results = await batch_embed_async(prompts, client)
    elapsed = time.perf_counter() - start

    print(f"\n=== Async Concurrent Calls ===")
    print(f"5 LLM calls in {elapsed:.2f}s (sequential would be {5*0.05:.2f}s)")

    print("\n=== Streaming Response ===")
    async for token in stream_response("tell me about RAG"):
        print(token, end="", flush=True)
    print()


def demo_context_manager():
    config = RAGConfig(temperature=0.0, similarity_threshold=0.8)
    print(f"\n=== Context Manager + Config ===")
    print(f"Strict mode: {config.is_strict_mode()}")
    with llm_session(config.llm_model) as session:
        session["calls"] += 3
        session["tokens_used"] += 1500
        print(f"Mid-session: {session}")


if __name__ == "__main__":
    demo_generators()
    asyncio.run(demo_async())
    demo_context_manager()
