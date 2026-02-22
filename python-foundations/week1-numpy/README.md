# Week 1 — NumPy Mastery

Operations that directly power AI/ML workloads: attention, embeddings, similarity search.

## What's Here

| File | Covers |
|------|--------|
| `numpy_mastery.py` | Vectors, matrix mul, broadcasting, SVD, performance comparison |

## Key Concepts

**Scaled dot-product attention** (transformer building block):
```python
scores = Q @ K.T / np.sqrt(d_model)   # similarity scores
weights = softmax(scores)              # attention distribution
output = weights @ V                   # weighted value aggregation
```

**Batch cosine similarity** (RAG retrieval):
```python
normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
scores = normalized @ query            # all 1000 docs at once, no loop
```

**Speedup**: vectorized ops are ~100-300x faster than Python loops for AI workloads.

## Run

```bash
python numpy_mastery.py
```
