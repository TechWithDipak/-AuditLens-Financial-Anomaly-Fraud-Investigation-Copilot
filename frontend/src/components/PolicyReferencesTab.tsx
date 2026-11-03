import React, { useState, useEffect } from 'react';
import {
  DocumentMetadata,
  RetrievedPassage,
  fetchPolicyCatalog,
  searchPolicies,
  reindexPolicies,
} from '../services/api';

const QUICK_SUGGESTIONS = [
  { label: 'Elder Financial Exploitation', query: 'elder financial exploitation sudden withdrawal of funds' },
  { label: 'Structuring & Smurfing', query: 'structuring smurfing transaction threshold repeated deposits' },
  { label: 'SAR 30-Day Deadline & Tipping Off', query: 'SAR filing timing 30 calendar days strict prohibition against tipping off' },
  { label: 'Ransomware & Crypto Hops', query: 'ransomware cyber extortion convertible virtual currency wallet hops' },
];

export const PolicyReferencesTab: React.FC = () => {
  const [query, setQuery] = useState('');
  const [issuingBody, setIssuingBody] = useState('');
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalog, setCatalog] = useState<DocumentMetadata[]>([]);
  const [passages, setPassages] = useState<RetrievedPassage[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchLatency, setSearchLatency] = useState<number | null>(null);
  const [disclaimer, setDisclaimer] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [reindexing, setReindexing] = useState(false);
  const [reindexMessage, setReindexMessage] = useState<string | null>(null);
  const [showCatalog, setShowCatalog] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function loadCatalog() {
      try {
        setCatalogLoading(true);
        const data = await fetchPolicyCatalog();
        if (isMounted) {
          setCatalog(data.documents);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to load policy catalog');
        }
      } finally {
        if (isMounted) {
          setCatalogLoading(false);
        }
      }
    }
    loadCatalog();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSearch = async (searchQuery?: string, filterBody?: string) => {
    const q = searchQuery !== undefined ? searchQuery : query;
    const bodyFilter = filterBody !== undefined ? filterBody : issuingBody;

    if (!q || q.trim().length < 2) {
      setError('Please enter at least 2 characters for policy search.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setReindexMessage(null);
      const res = await searchPolicies({
        query: q.trim(),
        issuing_body: bodyFilter ? bodyFilter : undefined,
        top_k: topK,
        min_similarity: 0.08,
      });
      setPassages(res.passages);
      setHasSearched(true);
      setSearchLatency(res.execution_time_ms);
      setDisclaimer(res.disclaimer);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setPassages([]);
      setHasSearched(true);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestionQuery: string) => {
    setQuery(suggestionQuery);
    handleSearch(suggestionQuery, issuingBody);
  };

  const handleClear = () => {
    setQuery('');
    setPassages([]);
    setHasSearched(false);
    setError(null);
    setSearchLatency(null);
  };

  const handleReindex = async () => {
    try {
      setReindexing(true);
      setReindexMessage(null);
      const res = await reindexPolicies();
      setReindexMessage(`✓ ${res.message}`);
      const refreshedCatalog = await fetchPolicyCatalog();
      setCatalog(refreshedCatalog.documents);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Reindexing failed');
    } finally {
      setReindexing(false);
    }
  };

  return (
    <div className="tab-pane active" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Title & Scope */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '0 0 0.5rem 0' }}>
              Regulatory Guidance &amp; Policy Citations (RAG)
            </h2>
            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.875rem', maxWidth: '800px' }}>
              Semantic passage retrieval over approved BSA/AML regulatory examination manuals, FinCEN advisories, and international FATF indicators. All citations preserve full source traceability and provenance.
            </p>
          </div>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setShowCatalog(!showCatalog)}
            style={{ fontSize: '0.85rem' }}
          >
            {showCatalog ? 'Hide Document Catalog ▲' : 'View Document Library (4) ▼'}
          </button>
        </div>

        {/* Collapsible Document Library */}
        {showCatalog && (
          <div style={{ marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0 }}>
                Approved Reference Documents ({catalog.length})
              </h3>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleReindex}
                disabled={reindexing}
                style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
              >
                {reindexing ? 'Re-indexing...' : 'Re-index Documents ⟳'}
              </button>
            </div>

            {reindexMessage && (
              <div style={{ color: '#10b981', fontSize: '0.85rem', marginBottom: '0.75rem', fontWeight: 500 }}>
                {reindexMessage}
              </div>
            )}

            {catalogLoading ? (
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Loading catalog...</p>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                {catalog.map((doc) => (
                  <div
                    key={doc.id}
                    style={{
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      padding: '1rem',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      gap: '0.5rem',
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                        <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8' }}>{doc.id}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{doc.publication_date}</span>
                      </div>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.25rem' }}>{doc.title}</div>
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.5rem' }}>{doc.issuing_body}</div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                        {doc.description}
                      </p>
                    </div>
                    <div style={{ paddingTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                      <a
                        href={doc.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ fontSize: '0.75rem', color: '#38bdf8', textDecoration: 'none' }}
                      >
                        Official Publication Source ↗
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Search Bar & Filters */}
      <div className="card">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
        >
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 300px' }}>
              <input
                type="text"
                className="form-control"
                placeholder="Search regulatory indicators, red flags, or SAR filing guidelines..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{ width: '100%', fontSize: '0.95rem', padding: '0.65rem 0.85rem' }}
              />
            </div>

            <div style={{ flex: '0 0 160px' }}>
              <select
                className="form-control"
                value={issuingBody}
                onChange={(e) => setIssuingBody(e.target.value)}
                style={{ width: '100%', height: '100%' }}
              >
                <option value="">All Issuing Bodies</option>
                <option value="FinCEN">FinCEN</option>
                <option value="FATF">FATF</option>
                <option value="FFIEC">FFIEC</option>
              </select>
            </div>

            <div style={{ flex: '0 0 110px' }}>
              <select
                className="form-control"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ width: '100%', height: '100%' }}
              >
                <option value={3}>Top 3 results</option>
                <option value={5}>Top 5 results</option>
                <option value={10}>Top 10 results</option>
              </select>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ minWidth: '110px' }}
            >
              {loading ? 'Searching...' : 'Search 🔍'}
            </button>

            {hasSearched && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleClear}
              >
                Clear
              </button>
            )}
          </div>

          {/* Quick Query Chips */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Suggested Queries:
            </span>
            {QUICK_SUGGESTIONS.map((item) => (
              <button
                key={item.label}
                type="button"
                onClick={() => handleSuggestionClick(item.query)}
                style={{
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  color: '#38bdf8',
                  borderRadius: '16px',
                  padding: '0.25rem 0.75rem',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  transition: 'background 0.2s',
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </form>

        {error && (
          <div style={{ marginTop: '1rem', color: '#f87171', fontSize: '0.85rem' }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Results Header & Mandatory Disclaimer */}
      {hasSearched && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {passages.length > 0
                ? `Retrieved ${passages.length} relevant passage${passages.length > 1 ? 's' : ''}`
                : 'No matching regulatory passages'}
              {searchLatency !== null && (
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginLeft: '0.5rem', fontWeight: 400 }}>
                  ({searchLatency} ms)
                </span>
              )}
            </span>
          </div>

          {/* Non-determination Regulatory Disclaimer */}
          <div
            style={{
              background: 'rgba(234, 179, 8, 0.08)',
              border: '1px solid rgba(234, 179, 8, 0.3)',
              borderRadius: '8px',
              padding: '0.85rem 1.25rem',
              display: 'flex',
              gap: '0.75rem',
              alignItems: 'center',
            }}
          >
            <span style={{ fontSize: '1.25rem' }}>⚖️</span>
            <div style={{ fontSize: '0.8rem', color: '#fde047', lineHeight: 1.4 }}>
              <strong>Advisory Decision-Support Notice:</strong> {disclaimer || (
                'Regulatory citations are provided for analyst decision support only and do not constitute legal determinations, regulatory findings, or automated SAR filing mandates.'
              )}
            </div>
          </div>
        </>
      )}

      {/* Passage Cards */}
      {hasSearched && passages.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {passages.map((passage, index) => {
            const matchPercent = Math.min(100, Math.round(passage.similarity_score * 100));
            return (
              <div
                key={passage.chunk_id || index}
                className="card"
                style={{
                  borderLeft: '4px solid #38bdf8',
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}
              >
                {/* Passage Top Bar */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                      <span className="status-badge normal" style={{ fontSize: '0.7rem' }}>
                        {passage.issuing_body}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {passage.publication_date}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: '#64748b' }}>
                        Chunk: {passage.chunk_id}
                      </span>
                    </div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: '0 0 0.25rem 0' }}>
                      {passage.document_title}
                    </h3>
                    <div style={{ fontSize: '0.825rem', color: '#94a3b8', fontWeight: 500 }}>
                      Section: {passage.section}
                    </div>
                  </div>

                  {/* Match Score Indicator */}
                  <div style={{ textAlign: 'right', minWidth: '120px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                      Relevance Match: <strong style={{ color: '#38bdf8' }}>{matchPercent}%</strong>
                    </div>
                    <div
                      style={{
                        height: '6px',
                        background: 'rgba(255, 255, 255, 0.1)',
                        borderRadius: '3px',
                        overflow: 'hidden',
                        width: '100%',
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          width: `${matchPercent}%`,
                          background: 'linear-gradient(90deg, #38bdf8, #818cf8)',
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Sanitized Passage Content */}
                <div
                  style={{
                    background: 'rgba(0, 0, 0, 0.25)',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    borderRadius: '6px',
                    padding: '1rem',
                    fontSize: '0.875rem',
                    lineHeight: 1.6,
                    color: '#e2e8f0',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {passage.text}
                </div>

                {/* Traceable Footer */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '0.5rem', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    Origin: {passage.document_id} &bull; Validated Passive Reference
                  </span>
                  <a
                    href={passage.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: '#38bdf8', textDecoration: 'none', fontWeight: 600 }}
                  >
                    View Official Reference ↗
                  </a>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State / No Relevant Sources */}
      {hasSearched && passages.length === 0 && !loading && (
        <div className="card" style={{ textAlign: 'center', padding: '3rem 2rem' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>🔍</div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            No Relevant Regulatory Source Found
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', maxWidth: '520px', margin: '0 auto 1.5rem', lineHeight: 1.5 }}>
            No indexed guidance exceeded the minimum relevance threshold for this query. AuditLens strictly prevents citation fabrication. Try searching for specific AML red flags, structuring terminology, or filing rules.
          </p>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => handleSuggestionClick('elder financial exploitation sudden withdrawal of funds')}
          >
            Try Example: Elder Financial Exploitation
          </button>
        </div>
      )}
    </div>
  );
};

