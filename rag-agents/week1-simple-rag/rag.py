"""
Simple RAG System: Document → Embed → Store → Retrieve → Generate
End-to-end RAG pipeline demonstrating the core loop.
"""

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Document Loading & Chunking ─────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    content: str
    source: str
    chunk_index: int
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    Sliding window chunking with overlap.
    Overlap prevents losing context at chunk boundaries.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size]
        if not window:
            break
        content = " ".join(window)
        chunk_id = hashlib.md5(f"{source}:{i}".encode()).hexdigest()[:8]
        chunks.append(Chunk(id=chunk_id, content=content, source=source, chunk_index=i))

    return chunks


# ─── Embedding (fake, deterministic) ─────────────────────────────────────────

def embed(text: str, dim: int = 128) -> list[float]:
    """Fake embedding — deterministic hash-based. Replace with real API in prod."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    vec = [math.sin((seed + i) * 0.0001) for i in range(dim)]
    norm = math.sqrt(sum(x ** 2 for x in vec))
    return [x / norm for x in vec]


def cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors already normalized


# ─── Vector Store ────────────────────────────────────────────────────────────

class SimpleVectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            chunk.embedding = embed(chunk.content)
        self.chunks.extend(chunks)
        print(f"Store: {len(self.chunks)} chunks indexed")

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_emb = embed(query)
        scored = [(c, cosine_sim(query_emb, c.embedding)) for c in self.chunks]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ─── Fake LLM ────────────────────────────────────────────────────────────────

def fake_llm_generate(system: str, context: str, question: str) -> str:
    """Simulates LLM generation from retrieved context. Replace with real API."""
    snippet = context[:200].replace("\n", " ")
    return (
        f"Based on the provided documents: {snippet}... "
        f"Therefore, the answer to '{question}' is contained in the retrieved context above."
    )


# ─── RAG Pipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, top_k: int = 3, min_score: float = 0.0):
        self.store = SimpleVectorStore()
        self.top_k = top_k
        self.min_score = min_score
        self.system_prompt = (
            "You are a helpful assistant. Answer questions based ONLY on the provided context. "
            "If the context doesn't contain enough information, say 'I don't know'."
        )

    def ingest(self, documents: list[dict]) -> None:
        """
        documents: [{"content": str, "source": str}]
        """
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc["content"], source=doc["source"])
            all_chunks.extend(chunks)
            print(f"  '{doc['source']}': {len(chunks)} chunks")

        self.store.add(all_chunks)

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        results = self.store.search(query, top_k=self.top_k)
        return [(c, s) for c, s in results if s >= self.min_score]

    def generate(self, query: str, results: list[tuple[Chunk, float]]) -> str:
        if not results:
            return "I don't have enough information to answer this question."

        context_parts = []
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source} (score: {score:.3f})]\n{chunk.content}"
            )
        context = "\n\n".join(context_parts)

        return fake_llm_generate(self.system_prompt, context, query)

    def ask(self, question: str) -> dict:
        print(f"\nQuestion: {question}")

        results = self.retrieve(question)
        answer = self.generate(question, results)

        sources = [
            {"source": c.source, "score": round(s, 4), "snippet": c.content[:100]}
            for c, s in results
        ]

        print(f"Retrieved: {len(results)} chunks")
        print(f"Answer: {answer[:150]}...")

        return {"question": question, "answer": answer, "sources": sources}


# ─── Demo ────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "source": "ai_engineering_guide",
        "content": """
        RAG (Retrieval-Augmented Generation) is a technique that combines a retrieval system
        with a language model. Instead of relying solely on the model's training data,
        RAG retrieves relevant documents from a knowledge base and uses them as context
        for the LLM to generate answers. This improves accuracy, reduces hallucination,
        and allows the system to answer questions about proprietary or recent data.

        The RAG pipeline has three main steps: (1) Indexing: documents are chunked and
        embedded into vectors stored in a vector database. (2) Retrieval: a user query is
        embedded and used to find the most similar document chunks via cosine similarity.
        (3) Generation: the retrieved chunks are passed as context to the LLM which
        generates a grounded answer.
        """,
    },
    {
        "source": "mlops_handbook",
        "content": """
        MLOps (Machine Learning Operations) is the practice of deploying and maintaining
        ML models in production reliably and efficiently. It includes: model versioning
        with tools like MLflow, automated retraining pipelines triggered by data drift,
        monitoring with Prometheus and Grafana, CI/CD pipelines that test and deploy models
        automatically, and A/B testing to compare model versions in production.

        Key MLOps metrics to track: model accuracy drift over time, data distribution shift,
        prediction latency (p50, p95, p99), error rates, and cost per prediction.
        """,
    },
    {
        "source": "vector_database_docs",
        "content": """
        Vector databases are specialized databases designed to store and query high-dimensional
        vectors (embeddings). Unlike traditional databases that use exact matching, vector DBs
        use approximate nearest neighbor (ANN) algorithms like HNSW or IVF to find the most
        similar vectors quickly. Popular options include pgvector (PostgreSQL extension),
        Pinecone (managed), Weaviate (open source), and Chroma (local development).

        pgvector supports cosine similarity, L2 distance, and inner product. For production
        RAG systems with millions of documents, use HNSW indexes for sub-millisecond retrieval.
        """,
    },
]

QUESTIONS = [
    "How does RAG reduce hallucination?",
    "What tools are used in MLOps for monitoring?",
    "What is HNSW and why is it used in vector databases?",
    "What happens if I ask something not in the documents?",
]

if __name__ == "__main__":
    print("=== Simple RAG System ===\n")

    rag = RAGPipeline(top_k=2, min_score=0.0)

    print("--- Ingesting Documents ---")
    rag.ingest(DOCUMENTS)

    print("\n--- Answering Questions ---")
    for q in QUESTIONS:
        result = rag.ask(q)
        print()
