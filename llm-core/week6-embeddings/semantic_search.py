"""
Semantic Search with Embeddings
Generates embeddings → stores vectors → cosine similarity search.
Foundation for RAG retrieval systems.
"""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    @staticmethod
    def from_text(content: str, **metadata) -> "Document":
        doc_id = hashlib.md5(content.encode()).hexdigest()[:8]
        return Document(id=doc_id, content=content, metadata=metadata)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


# ─── In-Memory Vector Store ───────────────────────────────────────────────────

class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    Production: replace with pgvector, Pinecone, or Weaviate.
    """

    def __init__(self):
        self._docs: dict[str, Document] = {}

    def add(self, doc: Document) -> None:
        if doc.embedding is None:
            raise ValueError(f"Document {doc.id} has no embedding")
        self._docs[doc.id] = doc

    def add_many(self, docs: list[Document]) -> None:
        for doc in docs:
            self.add(doc)

    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        if not self._docs:
            return []

        scores = []
        for doc in self._docs.values():
            score = cosine_similarity(query_embedding, doc.embedding)
            if score >= threshold:
                scores.append((doc, score))

        scores.sort(key=lambda x: -x[1])
        return [
            SearchResult(document=doc, score=round(score, 4), rank=i + 1)
            for i, (doc, score) in enumerate(scores[:top_k])
        ]

    def __len__(self) -> int:
        return len(self._docs)


# ─── Similarity Functions ─────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python — swap for numpy in production."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── Embedding Client ─────────────────────────────────────────────────────────

class EmbeddingClient:
    """
    Generates embeddings via OpenAI API.
    Falls back to a deterministic hash-based fake for offline use.
    """

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 384):
        self.model = model
        self.dim = dim
        self._cache: dict[str, list[float]] = {}
        self._use_real_api = bool(os.environ.get("OPENAI_API_KEY"))

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        if self._use_real_api:
            embedding = self._embed_real(text)
        else:
            embedding = self._embed_fake(text)

        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _embed_real(self, text: str) -> list[float]:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def _embed_fake(self, text: str) -> list[float]:
        """
        Deterministic fake embedding. NOT semantically meaningful.
        Used for testing pipeline logic without API keys.
        """
        import hashlib
        import math

        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        vec = []
        for i in range(self.dim):
            s = seed + i * 12345
            val = math.sin(s * 0.0001) * 0.5 + math.cos(s * 0.00007) * 0.5
            vec.append(val)

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]


# ─── Semantic Search System ───────────────────────────────────────────────────

class SemanticSearchSystem:
    def __init__(self, embedding_client: Optional[EmbeddingClient] = None):
        self.embedder = embedding_client or EmbeddingClient()
        self.store = VectorStore()

    def index(self, texts: list[str], **metadata) -> None:
        docs = [Document.from_text(t, **metadata) for t in texts]
        start = time.perf_counter()
        embeddings = self.embedder.embed_batch([d.content for d in docs])
        embed_time = (time.perf_counter() - start) * 1000

        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb

        self.store.add_many(docs)
        print(f"Indexed {len(docs)} documents in {embed_time:.1f}ms ({len(self.store)} total)")

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        query_emb = self.embedder.embed(query)
        return self.store.search(query_emb, top_k=top_k, threshold=threshold)

    def display_results(self, results: list[SearchResult]) -> None:
        if not results:
            print("  No results found.")
            return
        for r in results:
            snippet = r.document.content[:80].replace("\n", " ")
            print(f"  [{r.rank}] score={r.score:.4f} | {snippet}...")


# ─── Demo ────────────────────────────────────────────────────────────────────

CORPUS = [
    "Machine learning is the process of training models on data to make predictions.",
    "RAG combines retrieval systems with language models to answer questions from documents.",
    "Vector databases store embeddings and enable fast similarity search at scale.",
    "Fine-tuning adapts pre-trained language models to specific downstream tasks.",
    "The transformer architecture uses attention mechanisms to process sequences in parallel.",
    "Kubernetes orchestrates containerized applications across a cluster of machines.",
    "Prometheus collects metrics from services and stores them as time-series data.",
    "Gradient descent is the optimization algorithm used to train neural networks.",
    "LangChain provides tools for building LLM-powered applications with memory and tools.",
    "Embeddings represent text as dense vectors that capture semantic meaning.",
]

QUERIES = [
    "How do vector databases work?",
    "What is retrieval augmented generation?",
    "How do neural networks learn?",
    "How can I monitor my ML services?",
]

if __name__ == "__main__":
    system = SemanticSearchSystem()

    print("=== Indexing Corpus ===")
    system.index(CORPUS, source="ai_docs")

    print("\n=== Semantic Search Results ===")
    for query in QUERIES:
        print(f"\nQuery: '{query}'")
        results = system.search(query, top_k=3)
        system.display_results(results)
