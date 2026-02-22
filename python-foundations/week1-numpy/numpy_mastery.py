"""
NumPy Mastery: Vectors, Matrices, Linear Algebra for AI/ML
Covers operations used heavily in transformers and embeddings.
"""

import numpy as np
import time


# ─── Vector Operations ────────────────────────────────────────────────────────

def vector_ops_demo():
    """Core vector operations used in embeddings and similarity search."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    dot = np.dot(a, b)                          # similarity (pre-softmax)
    cosine_sim = dot / (np.linalg.norm(a) * np.linalg.norm(b))
    cross = np.cross(a, b)
    l2_norm = np.linalg.norm(a)

    print("=== Vector Operations ===")
    print(f"Dot product:     {dot}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Cross product:   {cross}")
    print(f"L2 norm:         {l2_norm:.4f}")
    return cosine_sim


# ─── Matrix Operations ────────────────────────────────────────────────────────

def matrix_ops_demo():
    """Matrix multiplication - the core of transformer attention."""
    # Simulate Q, K, V matrices from attention (batch=1, seq=4, dim=8)
    np.random.seed(42)
    seq_len, d_model = 4, 8
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)

    # Scaled dot-product attention (simplified)
    scores = Q @ K.T / np.sqrt(d_model)         # (seq, seq)
    scores_max = scores - scores.max(axis=-1, keepdims=True)  # numerical stability
    weights = np.exp(scores_max)
    weights /= weights.sum(axis=-1, keepdims=True)            # softmax
    attention_output = weights @ V              # (seq, d_model)

    print("\n=== Attention (Matrix Ops) ===")
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"Attention weights (row 0): {weights[0].round(3)}")
    print(f"Output shape: {attention_output.shape}")
    return attention_output


# ─── Broadcasting ─────────────────────────────────────────────────────────────

def broadcasting_demo():
    """Broadcasting - avoids loops, critical for batch processing."""
    # Normalize a batch of embeddings (used in RAG)
    embeddings = np.random.randn(100, 384)      # 100 docs, 384-dim embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # (100, 1)
    normalized = embeddings / norms             # broadcast: (100, 384)

    # Batch cosine similarity (query vs all docs)
    query = np.random.randn(384)
    query /= np.linalg.norm(query)
    similarities = normalized @ query           # (100,) — all at once, no loop

    top_k = np.argsort(similarities)[-5:][::-1]
    print("\n=== Broadcasting + Batch Similarity ===")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Top-5 doc indices: {top_k}")
    print(f"Top-5 scores:      {similarities[top_k].round(4)}")
    return similarities


# ─── Singular Value Decomposition ─────────────────────────────────────────────

def svd_demo():
    """SVD — used in PCA, LSA, low-rank approximations."""
    np.random.seed(0)
    A = np.random.randn(50, 20)               # 50 docs × 20 terms (TF-IDF style)

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Low-rank approximation (keep top k singular values)
    k = 5
    A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    reconstruction_error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)

    print("\n=== SVD / Low-Rank Approximation ===")
    print(f"Original shape: {A.shape}")
    print(f"Top-{k} singular values: {S[:k].round(2)}")
    print(f"Reconstruction error with k={k}: {reconstruction_error:.4f}")
    print(f"Compression ratio: {(k*(50+20+1)) / (50*20):.2f}x smaller")
    return reconstruction_error


# ─── Performance: Loops vs Vectorized ─────────────────────────────────────────

def performance_comparison():
    """Show why vectorization matters for AI workloads."""
    n = 1_000_000
    a = np.random.rand(n)
    b = np.random.rand(n)

    # Loop-based
    start = time.perf_counter()
    result_loop = sum(a[i] * b[i] for i in range(n))
    loop_time = time.perf_counter() - start

    # Vectorized
    start = time.perf_counter()
    result_vec = np.dot(a, b)
    vec_time = time.perf_counter() - start

    print("\n=== Performance: Loop vs Vectorized ===")
    print(f"Loop time:       {loop_time:.4f}s")
    print(f"Vectorized time: {vec_time:.4f}s")
    print(f"Speedup:         {loop_time / vec_time:.1f}x")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vector_ops_demo()
    matrix_ops_demo()
    broadcasting_demo()
    svd_demo()
    performance_comparison()
