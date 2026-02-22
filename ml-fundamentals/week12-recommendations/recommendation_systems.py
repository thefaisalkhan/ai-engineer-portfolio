"""
Phase 2 — Week 12: Recommendation Systems
==========================================
Covers: collaborative filtering, matrix factorization (SVD, ALS), content-based, hybrid
Job relevance: 61% of AI/ML roles; ubiquitous in e-commerce, streaming, social media
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from dataclasses import dataclass
from typing import Optional


@dataclass
class RecommendationResult:
    user_id: int
    recommended_items: list[int]
    scores: list[float]
    method: str


# ── Data Generation ──────────────────────────────────────────────────────────
def generate_ratings_matrix(
    n_users: int = 500,
    n_items: int = 200,
    density: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a sparse user-item ratings matrix (realistic: ~5% density).
    Sparsity is the fundamental challenge in collaborative filtering.
    """
    rng = np.random.default_rng(seed)
    n_ratings = int(n_users * n_items * density)

    user_ids = rng.integers(0, n_users, size=n_ratings)
    item_ids = rng.integers(0, n_items, size=n_ratings)
    ratings = rng.integers(1, 6, size=n_ratings).astype(float)

    df = pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "rating": ratings})
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    return df


# ── User-Based Collaborative Filtering ───────────────────────────────────────
class UserBasedCF:
    """
    User-based collaborative filtering: find similar users → recommend their items.
    "Users like you also liked..."
    """

    def __init__(self, n_neighbors: int = 20, min_common_items: int = 2):
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None

    def fit(self, ratings_df: pd.DataFrame):
        # Build user-item matrix (rows=users, cols=items)
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).fillna(0)

        # Normalize each user's ratings (mean-center to handle rating bias)
        matrix = self.user_item_matrix.values
        user_means = np.true_divide(
            matrix.sum(axis=1),
            (matrix != 0).sum(axis=1) + 1e-8,
        )
        matrix_centered = matrix - user_means[:, np.newaxis]
        matrix_centered[matrix == 0] = 0  # keep unrated as 0

        # Cosine similarity between users
        self.similarity_matrix = cosine_similarity(matrix_centered)
        np.fill_diagonal(self.similarity_matrix, 0)
        return self

    def recommend(self, user_id: int, n_recommendations: int = 10) -> RecommendationResult:
        if user_id not in self.user_item_matrix.index:
            return RecommendationResult(user_id, [], [], "UserBasedCF")

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similarities = self.similarity_matrix[user_idx]

        # Top-k neighbors
        top_neighbors = np.argsort(similarities)[-self.n_neighbors:][::-1]
        top_sims = similarities[top_neighbors]

        # Weighted sum of neighbor ratings
        neighbor_ratings = self.user_item_matrix.values[top_neighbors]
        weighted_ratings = (neighbor_ratings * top_sims[:, np.newaxis]).sum(axis=0)
        sim_sum = (np.abs(top_sims[:, np.newaxis]) * (neighbor_ratings != 0)).sum(axis=0)
        predicted = np.where(sim_sum > 0, weighted_ratings / sim_sum, 0)

        # Exclude already-rated items
        user_rated = self.user_item_matrix.values[user_idx] != 0
        predicted[user_rated] = -np.inf

        top_items_idx = np.argsort(predicted)[-n_recommendations:][::-1]
        items = self.user_item_matrix.columns[top_items_idx].tolist()
        scores = predicted[top_items_idx].tolist()

        return RecommendationResult(user_id, items, scores, "UserBasedCF")


# ── Matrix Factorization (SVD) ────────────────────────────────────────────────
class SVDRecommender:
    """
    Truncated SVD for matrix factorization.
    Decomposes the user-item matrix into latent factors:
    R ≈ U × Σ × V^T
    where U = user factors, V = item factors.
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.global_mean: float = 0.0

    def fit(self, ratings_df: pd.DataFrame):
        self.global_mean = ratings_df["rating"].mean()

        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).fillna(self.global_mean)

        # Sparse SVD
        matrix = self.user_item_matrix.values
        U, sigma, Vt = svds(csr_matrix(matrix - self.global_mean), k=min(self.n_factors, min(matrix.shape) - 1))

        self.user_factors = U * sigma
        self.item_factors = Vt.T
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_item_matrix.index:
            return self.global_mean
        if item_id not in self.user_item_matrix.columns:
            return self.global_mean
        u_idx = self.user_item_matrix.index.get_loc(user_id)
        i_idx = self.user_item_matrix.columns.get_loc(item_id)
        return float(self.global_mean + self.user_factors[u_idx] @ self.item_factors[i_idx])

    def recommend(self, user_id: int, n_recommendations: int = 10) -> RecommendationResult:
        if user_id not in self.user_item_matrix.index:
            return RecommendationResult(user_id, [], [], "SVD")

        u_idx = self.user_item_matrix.index.get_loc(user_id)
        all_scores = self.global_mean + self.user_factors[u_idx] @ self.item_factors.T

        # Exclude already-rated
        user_row = self.user_item_matrix.values[u_idx]
        already_rated = user_row != self.global_mean
        all_scores[already_rated] = -np.inf

        top_idx = np.argsort(all_scores)[-n_recommendations:][::-1]
        items = self.user_item_matrix.columns[top_idx].tolist()
        scores = all_scores[top_idx].tolist()

        return RecommendationResult(user_id, items, scores, "SVD-MatrixFactorization")


# ── Content-Based Filtering ───────────────────────────────────────────────────
class ContentBasedRecommender:
    """
    Content-based: recommend items similar in features to what user liked.
    Solves cold-start for new items (no ratings needed).
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.item_profiles: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None

    def fit(self, items_df: pd.DataFrame, text_column: str = "description"):
        self.item_ids = items_df.index.values
        tfidf_matrix = self.vectorizer.fit_transform(items_df[text_column])
        self.item_profiles = normalize(tfidf_matrix, norm="l2").toarray()
        return self

    def recommend_from_item(
        self,
        item_id: int,
        n_recommendations: int = 10,
    ) -> RecommendationResult:
        if item_id not in self.item_ids:
            return RecommendationResult(-1, [], [], "ContentBased")

        idx = np.where(self.item_ids == item_id)[0][0]
        similarities = cosine_similarity(
            self.item_profiles[idx:idx+1], self.item_profiles
        ).flatten()
        similarities[idx] = -1  # exclude self

        top_idx = np.argsort(similarities)[-n_recommendations:][::-1]
        items = self.item_ids[top_idx].tolist()
        scores = similarities[top_idx].tolist()

        return RecommendationResult(-1, items, scores, "ContentBased")


# ── Evaluation Metrics ────────────────────────────────────────────────────────
def precision_at_k(
    recommended: list,
    relevant: set,
    k: int,
) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / k


def recall_at_k(
    recommended: list,
    relevant: set,
    k: int,
) -> float:
    """Fraction of relevant items that appear in top-k."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / len(relevant)


def ndcg_at_k(
    recommended: list,
    relevant: set,
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain: rewards ranking relevant items higher."""
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_dcg = sum(
        1 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


if __name__ == "__main__":
    print("=== Generating Ratings Data ===")
    ratings_df = generate_ratings_matrix(n_users=200, n_items=100, density=0.08)
    print(f"Users: {ratings_df['user_id'].nunique()}, Items: {ratings_df['item_id'].nunique()}")
    print(f"Total ratings: {len(ratings_df)}, Sparsity: {1 - len(ratings_df) / (200*100):.1%}")

    print("\n=== User-Based Collaborative Filtering ===")
    ubcf = UserBasedCF(n_neighbors=15)
    ubcf.fit(ratings_df)
    result = ubcf.recommend(user_id=5, n_recommendations=10)
    print(f"Recommendations for user {result.user_id}: {result.recommended_items[:5]}")

    print("\n=== SVD Matrix Factorization ===")
    svd_rec = SVDRecommender(n_factors=20)
    svd_rec.fit(ratings_df)
    result = svd_rec.recommend(user_id=5, n_recommendations=10)
    print(f"SVD Recommendations for user {result.user_id}: {result.recommended_items[:5]}")

    # Spot prediction
    sample_item = ratings_df["item_id"].iloc[0]
    predicted_rating = svd_rec.predict(user_id=5, item_id=sample_item)
    print(f"Predicted rating for user=5, item={sample_item}: {predicted_rating:.3f}")

    print("\n=== Evaluation Metrics ===")
    recommended = [10, 25, 3, 47, 82, 15, 7, 33, 90, 4]
    relevant = {10, 3, 90, 7, 22, 56}
    print(f"Precision@5: {precision_at_k(recommended, relevant, k=5):.4f}")
    print(f"Recall@10:   {recall_at_k(recommended, relevant, k=10):.4f}")
    print(f"NDCG@10:     {ndcg_at_k(recommended, relevant, k=10):.4f}")
