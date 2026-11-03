"""Tests for AuditLensRetriever indexing, querying, and persistence."""

from pathlib import Path

from rag.retriever import AuditLensRetriever
from rag.schemas import PolicySearchRequest


def test_build_index_and_query_known_policies() -> None:
    """Retriever should index curated policies and return accurate citations."""
    retriever = AuditLensRetriever()
    count = retriever.build_index()
    assert count >= 10, f"Expected at least 10 chunks across policy documents, got {count}"

    # Query 1: Elder financial exploitation
    req = PolicySearchRequest(query="elder financial exploitation sudden withdrawal", top_k=3)
    res = retriever.search(req)

    assert res.has_results
    assert res.total_retrieved >= 1
    top_result = res.passages[0]
    assert "Elder Financial Exploitation" in top_result.document_title
    assert "FinCEN" in top_result.issuing_body
    assert top_result.similarity_score > 0.08
    assert top_result.source_url.startswith("https://")
    assert res.disclaimer != ""


def test_retrieval_with_issuing_body_filter() -> None:
    """Filter by issuing_body should restrict results to the specified authority."""
    retriever = AuditLensRetriever()
    retriever.load_index()

    # Query FATF explicitly
    req_fatf = PolicySearchRequest(
        query="structuring smurfing transaction threshold",
        issuing_body="FATF",
        top_k=5,
    )
    res_fatf = retriever.search(req_fatf)
    assert res_fatf.has_results
    for passage in res_fatf.passages:
        assert "FATF" in passage.issuing_body

    # Query FFIEC explicitly for SAR deadlines
    req_ffiec = PolicySearchRequest(
        query="SAR filing timing 30 calendar days tipping off",
        issuing_body="FFIEC",
        top_k=5,
    )
    res_ffiec = retriever.search(req_ffiec)
    assert res_ffiec.has_results
    for passage in res_ffiec.passages:
        assert "FFIEC" in passage.issuing_body


def test_retrieval_irrelevant_query_returns_no_results() -> None:
    """Irrelevant query below minimum similarity threshold must return empty results."""
    retriever = AuditLensRetriever()
    retriever.load_index()

    req = PolicySearchRequest(query="chocolate cake pastry baking recipe with strawberries", top_k=3)
    res = retriever.search(req)

    assert not res.has_results
    assert res.total_retrieved == 0
    assert res.passages == []


def test_retriever_save_and_reload(tmp_path: Path) -> None:
    """Retriever should save index to disk and reload without rebuilding."""
    store_dir = tmp_path / "vector_store"
    retriever = AuditLensRetriever(vector_store_dir=store_dir)

    indexed = retriever.build_index()
    assert indexed > 0
    assert (store_dir / "tfidf_vectorizer.joblib").is_file()
    assert (store_dir / "chunks.json").is_file()
    assert (store_dir / "index_meta.json").is_file()

    # Create new retriever pointing to same store
    reloaded_retriever = AuditLensRetriever(vector_store_dir=store_dir)
    loaded = reloaded_retriever.load_index()
    assert loaded

    res = reloaded_retriever.search(PolicySearchRequest(query="ransomware virtual currency cyber extortion"))
    assert res.has_results
    assert any("Ransomware" in p.document_title for p in res.passages)


def test_retriever_document_catalog() -> None:
    """Document catalog should return all 4 approved reference documents."""
    retriever = AuditLensRetriever()
    catalog = retriever.get_document_catalog()

    assert catalog.total_documents == 4
    doc_ids = {doc.id for doc in catalog.documents}
    assert "FIN-2021-A003" in doc_ids
    assert "FATF-2023-RED-FLAGS" in doc_ids
    assert "FFIEC-BSA-AML-SARS" in doc_ids
    assert "FIN-2020-A001" in doc_ids

