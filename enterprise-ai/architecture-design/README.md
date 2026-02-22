# Enterprise AI System Architecture

Multi-service AI platform design. Documents: component breakdown, data flows, latency budgets, cost models, and architectural decision records (ADRs).

---

## System Overview

```
                           ┌─────────────────┐
                           │   API Gateway    │
                           │  (rate limit,    │
                           │   auth, routing) │
                           └────────┬────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
    ┌─────────▼────────┐  ┌─────────▼────────┐  ┌─────────▼────────┐
    │  Ingestion Svc   │  │    RAG Service   │  │  Agent Service   │
    │  (data intake,   │  │ (query, retrieve,│  │ (tool calling,   │
    │   validation,    │  │  generate)       │  │  planning)       │
    │   versioning)    │  └─────────┬────────┘  └─────────┬────────┘
    └─────────┬────────┘            │                      │
              │                     │                      │
    ┌─────────▼────────┐  ┌─────────▼────────┐  ┌─────────▼────────┐
    │ Embedding Service│  │  Vector DB       │  │    LLM APIs      │
    │ (batch embed,    │  │  (pgvector,      │  │  (OpenAI /       │
    │  caching,        │  │   HNSW index)    │  │   Anthropic)     │
    │  model mgmt)     │  └──────────────────┘  └──────────────────┘
    └──────────────────┘
              │
    ┌─────────▼────────────────────────────────────────────────┐
    │  Infrastructure                                          │
    │  PostgreSQL │ Redis (cache) │ Celery (async) │ S3 (docs) │
    └──────────────────────────────────────────────────────────┘
              │
    ┌─────────▼────────────────────────────────────────────────┐
    │  Observability                                           │
    │  Prometheus │ Grafana │ OpenTelemetry │ AlertManager     │
    └──────────────────────────────────────────────────────────┘
```

---

## Latency Budget (500ms Total)

| Component | Budget | Why |
|-----------|--------|-----|
| API Gateway (auth + routing) | 10ms | JWT verify, rate check |
| Query embedding | 30ms | text-embedding-3-small, cached |
| Vector retrieval (top-5) | 20ms | HNSW index, pgvector |
| Reranking | 40ms | cross-encoder, optional |
| LLM generation | 350ms | gpt-4o-mini, streaming |
| Response serialization | 10ms | JSON encoding |
| **Total** | **~460ms** | **p95 target: < 500ms** |

---

## Cost Model (per 1000 requests)

| Component | Cost | Notes |
|-----------|------|-------|
| Query embedding | $0.002 | 500 tokens avg, text-embedding-3-small |
| Vector search | $0.001 | pgvector, compute cost |
| LLM generation | $0.15 | gpt-4o-mini, 300 in + 500 out tokens |
| **Total** | **~$0.15** | **$150/million requests** |

Optimization: cache popular queries (30% hit rate → $105/million).

---

## Scaling Strategy

| Traffic | Architecture | Cost/month |
|---------|-------------|-----------|
| < 100 RPS | Single instance, managed DB | ~$200 |
| 100-1000 RPS | k8s, HPA, read replicas | ~$1,500 |
| 1000+ RPS | Multi-region, CDN, Kafka | ~$10,000+ |

---

## Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|---------|
| LLM API down | Health check + circuit breaker | Fallback to secondary model |
| Vector DB overload | p95 latency spike | Connection pool limit, queue |
| Embedding model slow | Latency alert | Cached embeddings, degraded mode |
| Data drift | Statistical tests | Alert + trigger retraining |
| Cost spike | Budget alert | Rate limit per user |

---

## Architectural Decision Records (ADRs)

### ADR-001: pgvector over Pinecone

**Decision**: Use pgvector (PostgreSQL extension) instead of managed vector DB.

**Rationale**:
- Single database for both structured data and vectors (reduces ops complexity)
- HNSW indexes provide sub-10ms retrieval at 1M+ vectors
- No additional vendor dependency or cost ($0 vs ~$70/month for Pinecone starter)
- SQL joins between metadata and vectors work natively

**Tradeoff**: Manual index tuning required; Pinecone scales more easily above 100M vectors.

---

### ADR-002: gpt-4o-mini as default model

**Decision**: Use gpt-4o-mini as the default generation model, with gpt-4o as fallback.

**Rationale**:
- 96% cheaper than gpt-4o for equivalent quality on RAG tasks
- RAG provides context, reducing need for model's parametric memory
- p95 latency: 200ms vs 800ms for gpt-4o

**Tradeoff**: Worse on complex multi-hop reasoning; switch to gpt-4o for those cases.

---

### ADR-003: Celery for async embedding

**Decision**: Use Celery + Redis for background document embedding.

**Rationale**:
- Document ingestion is async — user shouldn't wait for embeddings
- Celery provides retry logic, task queuing, and visibility
- Redis is already in stack for caching

**Tradeoff**: Additional operational complexity; simpler for small scale.
