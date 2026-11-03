import React, { useState, useRef } from 'react';
import { uploadDataset, UploadResult } from '../services/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (result: UploadResult) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUploadSuccess }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [ingestToDb, setIngestToDb] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a .csv file to upload.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await uploadDataset(file, isTraining, ingestToDb);
      setResult(data);
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="upload-title">
      <div className="modal-content">
        <div className="modal-header">
          <h2 id="upload-title" className="modal-title">Upload &amp; Validate Transaction Dataset</h2>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {error && (
              <div className="banner" style={{ borderLeftColor: 'var(--status-danger)', backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
                <div className="banner-title" style={{ color: '#f87171' }}>Upload or Validation Error</div>
                <div className="banner-body">{error}</div>
              </div>
            )}

            {result && (
              <div className="banner" style={{ borderLeftColor: 'var(--status-success)', backgroundColor: 'rgba(16, 185, 129, 0.1)' }}>
                <div className="banner-title" style={{ color: '#34d399' }}>Dataset Validated Successfully!</div>
                <div className="banner-body">
                  <strong>{result.filename}</strong>: {result.row_count} rows, {result.column_count} features.
                  {result.ingested_count > 0 && (
                    <div>Ingested <strong>{result.ingested_count}</strong> transactions into database with automated scoring.</div>
                  )}
                  {result.duplicate_count > 0 && (
                    <div style={{ color: '#fbbf24', marginTop: '0.25rem' }}>
                      Notice: {result.duplicate_count} duplicate row(s) identified.
                    </div>
                  )}
                </div>
              </div>
            )}

            <div
              className="dropzone"
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  fileInputRef.current?.click();
                }
              }}
              tabIndex={0}
              role="button"
              aria-label="Select CSV file to upload"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              <div className="dropzone-icon">&#128196;</div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                {file ? file.name : 'Click to select CSV file'}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Supports standard Credit Card Fraud transaction schema (.csv)'}
              </div>
            </div>

            <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={ingestToDb}
                  onChange={(e) => setIngestToDb(e.target.checked)}
                />
                <span>Ingest valid rows into local SQLite database for immediate analyst triage &amp; scoring</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={isTraining}
                  onChange={(e) => setIsTraining(e.target.checked)}
                />
                <span>Enforce training mode (requires ground truth <code>Class</code> column)</span>
              </label>
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn btn-outline" onClick={onClose} disabled={loading}>
              {result ? 'Done' : 'Cancel'}
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading || !file}>
              {loading ? 'Validating & Ingesting...' : 'Upload & Validate'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

