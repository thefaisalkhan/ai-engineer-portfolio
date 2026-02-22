"""
Production RAG Service
Handles: query embedding → hybrid retrieval → reranking → LLM generation → streaming.
"""

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


# ─── Schemas ──────────────────────────────────────────────────────────────────

@dataclass
class QueryRequest:
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.60
    stream: bool = False


@dataclass
class Source:
    doc_id: str
    source: str
    snippet: str
    score: float
    retrieval_method: str  # "semantic" | "bm25" | "hybrid"


@dataclass
class RAGResponse:
    request_id: str
    answer: str
    sources: list[Source]
    confidence: float
    latency_ms: float
    tokens_used: int
    fallback_used: bool = False


# ─── Retrieval ────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines semantic (vector) + BM25 (keyword) retrieval.
    Hybrid consistently outperforms either alone: handles both semantic similarity
    and exact keyword matches (e.g., product IDs, technical terms).
    """

    def __init__(self, documents: list[dict], alpha: float = 0.7):
        self.documents = documents
        self.alpha = alpha  # weight: alpha * semantic + (1-alpha) * bm25

    def _embed(self, text: str, dim: int = 64) -> list[float]:
        seed = int(hashlib.sha256(text.lower().encode()).hexdigest(), 16)
        vec = [math.sin((seed + i) * 0.0001) for i in range(dim)]
        norm = math.sqrt(sum(x ** 2 for x in vec))
        return [x / norm for x in vec]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _bm25_score(self, query: str, doc: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Simplified BM25 scoring."""
        query_terms = query.lower().split()
        doc_terms = doc.lower().split()
        doc_len = len(doc_terms)
        avg_doc_len = 50  # assume average
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            idf = math.log(1 + (len(self.documents) + 0.5) / (1 + 1))  # simplified
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float, str]]:
        query_emb = self._embed(query)

        scored = []
        for doc in self.documents:
            sem_score = self._cosine(query_emb, self._embed(doc["content"]))
            bm25_score = self._bm25_score(query, doc["content"])
            bm25_norm = min(bm25_score / 5.0, 1.0)  # normalize BM25 to [0,1]
            hybrid = self.alpha * sem_score + (1 - self.alpha) * bm25_norm
            scored.append((doc, hybrid, "hybrid"))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ─── Reranker ────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks retrieved documents by relevance to query.
    Cross-encoder is more accurate than bi-encoder but slower — use only on top-K.
    """

    def rerank(self, query: str, candidates: list[tuple[dict, float, str]]) -> list[tuple[dict, float, str]]:
        # Simulate cross-encoder scoring (in prod: use sentence-transformers)
        reranked = []
        for doc, score, method in candidates:
            # Boost score for keyword overlap (simple proxy for cross-encoder)
            query_words = set(query.lower().split())
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words & doc_words) / max(len(query_words), 1)
            new_score = score * 0.7 + overlap * 0.3
            reranked.append((doc, round(new_score, 4), "hybrid+reranked"))

        reranked.sort(key=lambda x: -x[1])
        return reranked


# ─── Generation ──────────────────────────────────────────────────────────────

class LLMGenerator:
    """Generates answers from retrieved context. Replace with real API call."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, question: str, context: str) -> tuple[str, int]:
        # Simulate generation
        snippet = context[:200].replace("\n", " ")
        answer = (
            f"Based on the retrieved documents, {snippet}... "
            f"In summary, the answer to '{question[:50]}' is derived from the provided context."
        )
        tokens = len(question.split()) + len(context.split()) + len(answer.split())
        return answer, tokens

    def detect_hallucination(self, answer: str, context: str) -> float:
        """Check what fraction of answer words appear in context — simple grounding check."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        return len(answer_words & context_words) / max(len(answer_words), 1)


# ─── Conversation Memory ──────────────────────────────────────────────────────

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.sessions: dict[str, list[dict]] = {}
        self.max_turns = max_turns

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        # Keep only recent turns
        if len(self.sessions[session_id]) > self.max_turns * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns * 2:]

    def get_context(self, session_id: str) -> str:
        history = self.sessions.get(session_id, [])
        return "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])  # last 2 turns


# ─── RAG Service ──────────────────────────────────────────────────────────────

class RAGService:
    def __init__(self, documents: list[dict]):
        self.retriever = HybridRetriever(documents)
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()
        self.memory = ConversationMemory()

    def query(self, req: QueryRequest) -> RAGResponse:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]

        # 1. Retrieve
        candidates = self.retriever.retrieve(req.question, top_k=req.top_k * 2)

        # 2. Rerank
        reranked = self.reranker.rerank(req.question, candidates)

        # 3. Filter by threshold
        filtered = [(doc, score, method) for doc, score, method in reranked if score >= req.min_score]

        # 4. Guardrail: no relevant docs
        fallback_used = False
        if not filtered:
            filtered = reranked[:1]  # use best even if below threshold
            fallback_used = True

        # 5. Build context
        context_parts = []
        for doc, score, _ in filtered[:req.top_k]:
            context_parts.append(f"[{doc.get('source', 'unknown')}]\n{doc['content']}")
        context = "\n\n".join(context_parts)

        # 6. Add conversation history
        if req.session_id:
            history = self.memory.get_context(req.session_id)
            if history:
                context = f"Previous conversation:\n{history}\n\n---\n\n{context}"

        # 7. Generate
        answer, tokens = self.generator.generate(req.question, context)

        # 8. Hallucination check
        grounding_score = self.generator.detect_hallucination(answer, context)

        # 9. Update memory
        if req.session_id:
            self.memory.add(req.session_id, "user", req.question)
            self.memory.add(req.session_id, "assistant", answer)

        sources = [
            Source(
                doc_id=hashlib.md5(doc["content"].encode()).hexdigest()[:8],
                source=doc.get("source", "unknown"),
                snippet=doc["content"][:100] + "...",
                score=score,
                retrieval_method=method,
            )
            for doc, score, method in filtered[:req.top_k]
        ]

        return RAGResponse(
            request_id=request_id,
            answer=answer,
            sources=sources,
            confidence=round(grounding_score, 3),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            tokens_used=tokens,
            fallback_used=fallback_used,
        )


# ─── Demo ────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {"source": "architecture_guide", "content": "The system uses hybrid retrieval combining BM25 keyword search with semantic vector search. Alpha=0.7 weights semantic higher."},
    {"source": "operations_manual", "content": "For p95 latency above 500ms, scale the embedding service horizontally. Check Redis cache hit rate first — should be above 40%."},
    {"source": "cost_guide", "content": "Average cost per query is $0.00015. Caching reduces this by 30%. Use gpt-4o-mini for most queries, reserve gpt-4o for complex reasoning."},
    {"source": "troubleshooting", "content": "If retrieval quality drops, check embedding model version and vector index freshness. Re-index if documents were updated more than 24 hours ago."},
    {"source": "security_policy", "content": "All user data is anonymized before LLM processing. Prompt injection detected via pattern matching and refused with 400 status code."},
]

if __name__ == "__main__":
    service = RAGService(KNOWLEDGE_BASE)
    session = "demo-session-1"

    questions = [
        "How does the retrieval system work?",
        "What should I do if latency is too high?",
        "How much does each query cost?",
        "Is user data sent to OpenAI?",
    ]

    for q in questions:
        req = QueryRequest(question=q, session_id=session, top_k=3)
        resp = service.query(req)
        print(f"\nQ: {q}")
        print(f"A: {resp.answer[:150]}...")
        print(f"   Sources: {[s.source for s in resp.sources]}")
        print(f"   Confidence: {resp.confidence} | Latency: {resp.latency_ms}ms | Tokens: {resp.tokens_used}")
        if resp.fallback_used:
            print("   [FALLBACK: low relevance, used best available]")
