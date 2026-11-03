"""Retriever implementation for AuditLens RAG using TF-IDF and Cosine Similarity."""

import json
import logging
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunker import chunk_document
from rag.sanitizer import sanitize_document_text
from rag.schemas import (
    DocumentCatalogResponse,
    DocumentChunk,
    DocumentMetadata,
    PolicySearchRequest,
    PolicySearchResponse,
    RetrievedPassage,
)

logger = logging.getLogger("auditlens.rag.retriever")


class AuditLensRetriever:
    """TF-IDF and Cosine Similarity Vector Retriever for approved regulatory documents."""

    def __init__(
        self,
        policies_dir: Path | str = Path("data/policies"),
        vector_store_dir: Path | str = Path("data/vector_store"),
    ) -> None:
        self.policies_dir = Path(policies_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.vectorizer: TfidfVectorizer | None = None
        self.chunks: list[DocumentChunk] = []
        self.chunk_matrix = None
        self._is_indexed = False

    def build_index(self) -> int:
        """Ingest all policy documents from policies_dir and build the vector index.

        Returns:
            Total number of chunks indexed.
        """
        manifest_path = self.policies_dir / "manifest.json"
        if not manifest_path.is_file():
            logger.warning(f"Policy manifest not found at {manifest_path}")
            return 0

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        documents_meta = [DocumentMetadata(**doc) for doc in manifest.get("documents", [])]
        all_chunks: list[DocumentChunk] = []

        for doc_meta in documents_meta:
            doc_file = self.policies_dir / doc_meta.filename
            if not doc_file.is_file():
                logger.warning(f"Referenced policy file {doc_file} not found; skipping.")
                continue

            try:
                chunks = chunk_document(doc_file, doc_meta)
                all_chunks.extend(chunks)
                logger.info(f"Ingested {len(chunks)} chunks from {doc_meta.id} ({doc_meta.title})")
            except (ValueError, OSError, KeyError) as e:
                logger.error(f"Failed to chunk document {doc_file}: {e}")

        if not all_chunks:
            logger.warning("No document chunks extracted. Vector store index is empty.")
            self.chunks = []
            self.vectorizer = None
            self.chunk_matrix = None
            self._is_indexed = False
            return 0

        # Build TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=10000,
        )
        texts = [chunk.text for chunk in all_chunks]
        matrix = vectorizer.fit_transform(texts)

        # Persist artifacts to vector_store_dir
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        vectorizer_path = self.vector_store_dir / "tfidf_vectorizer.joblib"
        chunks_path = self.vector_store_dir / "chunks.json"
        meta_path = self.vector_store_dir / "index_meta.json"

        joblib.dump(vectorizer, vectorizer_path)
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump([chunk.model_dump() for chunk in all_chunks], f, indent=2)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_chunks": len(all_chunks),
                    "total_documents": len(documents_meta),
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                f,
                indent=2,
            )

        self.vectorizer = vectorizer
        self.chunks = all_chunks
        self.chunk_matrix = matrix
        self._is_indexed = True

        logger.info(
            f"Successfully built vector store with {len(all_chunks)} chunks across "
            f"{len(documents_meta)} documents."
        )
        return len(all_chunks)

    def load_index(self) -> bool:
        """Load the persisted index from vector_store_dir, or build it if missing.

        Returns:
            True if an active index is ready for queries, False otherwise.
        """
        vectorizer_path = self.vector_store_dir / "tfidf_vectorizer.joblib"
        chunks_path = self.vector_store_dir / "chunks.json"

        if not vectorizer_path.is_file() or not chunks_path.is_file():
            logger.info("Persisted index not found. Building index from policies directory...")
            indexed_count = self.build_index()
            return indexed_count > 0

        try:
            self.vectorizer = joblib.load(vectorizer_path)
            with chunks_path.open("r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            self.chunks = [DocumentChunk(**chunk) for chunk in raw_chunks]
            if self.chunks and self.vectorizer is not None:
                texts = [c.text for c in self.chunks]
                self.chunk_matrix = self.vectorizer.transform(texts)
                self._is_indexed = True
                return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load vector store from disk ({e}). Attempting rebuild...")
            indexed_count = self.build_index()
            return indexed_count > 0

        return False

    def ensure_index_ready(self) -> None:
        """Ensure the vector index is loaded or constructed before querying."""
        if not self._is_indexed or self.vectorizer is None or self.chunk_matrix is None:
            self.load_index()

    def search(self, request: PolicySearchRequest) -> PolicySearchResponse:
        """Search policy references using cosine similarity on TF-IDF vectors.

        Args:
            request: PolicySearchRequest with query, top_k, min_similarity, and issuing_body filter.

        Returns:
            PolicySearchResponse with retrieved passages and execution latency.
        """
        start_time = time.perf_counter()
        self.ensure_index_ready()

        if not self._is_indexed or self.vectorizer is None or self.chunk_matrix is None or not self.chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PolicySearchResponse(
                query=request.query,
                total_retrieved=0,
                passages=[],
                has_results=False,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # 1. Sanitize query
        sanitized_query = sanitize_document_text(request.query).sanitized_text.strip()
        if not sanitized_query:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PolicySearchResponse(
                query=request.query,
                total_retrieved=0,
                passages=[],
                has_results=False,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # 2. Vectorize query and compute similarity
        query_vec = self.vectorizer.transform([sanitized_query])
        similarities = cosine_similarity(query_vec, self.chunk_matrix)[0]

        # 3. Filter candidates
        candidates: list[tuple[int, float]] = []
        for idx, score in enumerate(similarities):
            if score < request.min_similarity:
                continue

            chunk = self.chunks[idx]
            if request.issuing_body:
                # Case-insensitive substring match on issuing body
                req_issuing = request.issuing_body.strip().lower()
                chunk_issuing = chunk.issuing_body.lower()
                if req_issuing not in chunk_issuing:
                    continue

            candidates.append((idx, float(score)))

        # 4. Sort descending by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[: request.top_k]

        # 5. Build passages
        passages: list[RetrievedPassage] = []
        for idx, score in top_candidates:
            chunk = self.chunks[idx]
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    issuing_body=chunk.issuing_body,
                    section=chunk.section,
                    source_url=chunk.source_url,
                    publication_date=chunk.publication_date,
                    text=chunk.text,
                    similarity_score=round(score, 4),
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return PolicySearchResponse(
            query=request.query,
            total_retrieved=len(passages),
            passages=passages,
            has_results=len(passages) > 0,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def get_document_catalog(self) -> DocumentCatalogResponse:
        """Retrieve manifest metadata for all approved reference documents."""
        manifest_path = self.policies_dir / "manifest.json"
        if not manifest_path.is_file():
            return DocumentCatalogResponse(version="1.0.0", total_documents=0, documents=[])

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        docs = [DocumentMetadata(**doc) for doc in manifest.get("documents", [])]
        return DocumentCatalogResponse(
            version=manifest.get("version", "1.0.0"),
            total_documents=len(docs),
            documents=docs,
        )
