"""Vector Indexing and Cosine Similarity Retrieval Engine."""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.rag import RagChunk, RagCitation


class VectorIndex:
    """In-memory and persisted TF-IDF vector index for regulatory knowledge retrieval."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True,
        )
        self.chunks: List[RagChunk] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def build_index(self, chunks: List[RagChunk]) -> None:
        """Fit vectorizer on all chunk texts and store representation."""
        if not chunks:
            self.chunks = []
            self.tfidf_matrix = None
            self.is_indexed = False
            return

        self.chunks = chunks
        corpus = [f"{c.title} {c.section} {c.text}" for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
        issuing_body_filter: Optional[str] = None,
    ) -> List[RagCitation]:
        """Retrieve top-k relevant citations, applying metadata filter and weak-similarity cutoff."""
        if not self.is_indexed or self.tfidf_matrix is None or len(self.chunks) == 0:
            return []

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Rank candidates by descending similarity
        ranked_indices = np.argsort(similarities)[::-1]

        results: List[RagCitation] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_similarity:
                break  # Stop when similarity drops below threshold

            chunk = self.chunks[idx]

            # Apply issuing body filter if requested
            if (
                issuing_body_filter
                and issuing_body_filter.lower() not in chunk.issuing_body.lower()
            ):
                continue

            results.append(
                RagCitation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    issuing_body=chunk.issuing_body,
                    section=chunk.section,
                    source_url=chunk.source_url,
                    similarity_score=round(score, 4),
                    text=chunk.text,
                    has_injection_risk=chunk.has_injection_risk,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save(self, file_path: Path) -> None:
        """Persist vector index and chunks to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "chunks": self.chunks,
                "tfidf_matrix": self.tfidf_matrix,
                "is_indexed": self.is_indexed,
            },
            file_path,
        )

    def load(self, file_path: Path) -> bool:
        """Load vector index and chunks from disk with safe corruption recovery."""
        if not file_path.exists():
            return False
        try:
            data = joblib.load(file_path)
            self.vectorizer = data["vectorizer"]
            self.chunks = data["chunks"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.is_indexed = data["is_indexed"]
            return True
        except Exception:
            # Corrupted index file, reset state and signal reload required
            self.chunks = []
            self.tfidf_matrix = None
            self.is_indexed = False
            return False
