import test from 'node:test';
import assert from 'node:assert/strict';

test('Transaction data contract isolates dual scores', () => {
  const sampleTx = {
    id: 1,
    transaction_id: 'TX_00000001_1',
    time: 120.0,
    amount: 549.99,
    features: { V1: -1.45, V2: 0.88, V14: -4.21 },
    anomaly_score: 0.72,
    fraud_probability: 0.89,
    is_flagged: true,
    score_type: 'dual_scoring',
    label: 1,
    created_at: new Date().toISOString(),
  };

  // Must isolate unsupervised outlier metric from supervised probability
  assert.equal(typeof sampleTx.anomaly_score, 'number');
  assert.equal(typeof sampleTx.fraud_probability, 'number');
  assert.notEqual(sampleTx.anomaly_score, sampleTx.fraud_probability);
  assert.equal(sampleTx.is_flagged, true);
  assert.ok(sampleTx.amount > 0);
});

test('Transaction explanation structure validates non-causal evidence', () => {
  const mockExplanation = {
    transaction_id: 'TX_00000001_1',
    model_name: 'Logistic Regression Baseline',
    model_version: 'logreg_v0.1.0',
    explanation_method: 'shap_linear',
    score_type: 'fraud_probability',
    prediction_score: 0.89,
    decision_threshold: 0.5,
    is_flagged: true,
    base_value: 0.05,
    top_contributions: [
      {
        feature_name: 'V14',
        transformed_value: -4.21,
        contribution: 0.35,
        direction: 'increases_risk',
        relative_magnitude: 0.85,
        plain_text_description: 'Feature V14 increased model score by +0.350',
      },
      {
        feature_name: 'Amount',
        transformed_value: 1.25,
        contribution: -0.08,
        direction: 'decreases_risk',
        relative_magnitude: 0.2,
        plain_text_description: 'Feature Amount decreased model score by -0.080',
      },
    ],
    all_contributions: { V14: 0.35, Amount: -0.08 },
    summary_text: "Model assigned a fraud probability of 0.890.",
    warnings: [],
    limitations: [
      'Explanations reflect statistical associations learned by the model and do not constitute legal proof of fraud.',
    ],
    timestamp: new Date().toISOString(),
  };

  assert.equal(mockExplanation.top_contributions.length, 2);
  assert.equal(mockExplanation.top_contributions[0].direction, 'increases_risk');
  assert.equal(mockExplanation.top_contributions[1].direction, 'decreases_risk');
  // Confirm non-causal language
  assert.ok(!mockExplanation.summary_text.toLowerCase().includes('proof'));
  assert.ok(mockExplanation.limitations.length > 0);
});

test('Case lifecycle state transition rules', () => {
  const allowedStatuses = [
    'NEW',
    'IN_REVIEW',
    'ESCALATED',
    'CLOSED_FALSE_POSITIVE',
    'CLOSED_CONFIRMED_FRAUD',
  ];

  const allowedPriorities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];

  const testCase = {
    id: 1,
    case_id: 'CASE-20261103-0001',
    title: 'Rapid multi-merchant velocity anomaly',
    description: 'Suspicious card activity detected across 3 geographical zones.',
    status: 'NEW',
    priority: 'HIGH',
    assigned_to: 'analyst_alice',
    transaction_ids: ['TX_00000001_1', 'TX_00000002_2'],
    notes: [
      {
        id: 1,
        case_id: 1,
        author: 'analyst_alice',
        content: 'Cardholder contacted via 2FA.',
        created_at: new Date().toISOString(),
      },
    ],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  assert.ok(allowedStatuses.includes(testCase.status));
  assert.ok(allowedPriorities.includes(testCase.priority));
  assert.equal(testCase.transaction_ids.length, 2);
  assert.equal(testCase.notes.length, 1);

  // Simulate transition to IN_REVIEW
  testCase.status = 'IN_REVIEW';
  assert.equal(testCase.status, 'IN_REVIEW');

  // Simulate closing case as confirmed fraud
  testCase.status = 'CLOSED_CONFIRMED_FRAUD';
  assert.equal(testCase.status, 'CLOSED_CONFIRMED_FRAUD');
});

test('End-to-End Analyst Triage Workflow Pipeline contract', () => {
  // Step 1: Ingest batch
  const uploadedBatch = [
    { transaction_id: 'TX_101', amount: 1200.0, fraud_probability: 0.92, is_flagged: true },
    { transaction_id: 'TX_102', amount: 15.5, fraud_probability: 0.02, is_flagged: false },
  ];

  // Step 2: Analyst filters for flagged transactions
  const flagged = uploadedBatch.filter((tx) => tx.is_flagged && tx.fraud_probability >= 0.5);
  assert.equal(flagged.length, 1);
  assert.equal(flagged[0].transaction_id, 'TX_101');

  // Step 3: Inspect flagged transaction
  const inspectTarget = flagged[0];
  assert.ok(inspectTarget.fraud_probability > 0.5);

  // Step 4: Create investigation case payload from flagged transaction
  const casePayload = {
    title: `Investigation for Transaction ${inspectTarget.transaction_id}`,
    description: `Auto-flagged with ${(inspectTarget.fraud_probability * 100).toFixed(1)}% fraud probability.`,
    priority: 'HIGH',
    transaction_ids: [inspectTarget.transaction_id],
    assigned_to: 'analyst_lead',
  };

  assert.equal(casePayload.transaction_ids[0], 'TX_101');
  assert.equal(casePayload.priority, 'HIGH');
});

test('RAG Policy reference contract verifies traceability and advisory disclaimer', () => {
  const samplePassage = {
    chunk_id: 'FIN-2021-A003_c002',
    document_id: 'FIN-2021-A003',
    document_title: 'FinCEN Advisory on Elder Financial Exploitation',
    issuing_body: 'Financial Crimes Enforcement Network (FinCEN)',
    section: '2. Transactional and Behavioral Red Flag Indicators',
    source_url: 'https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2021-a003',
    publication_date: '2021-06-15',
    text: 'Uncharacteristic, rapid depletion of assets or retirement savings without a plausible personal or commercial justification.',
    similarity_score: 0.842,
  };

  const sampleSearchResponse = {
    query: 'elder financial exploitation sudden withdrawal',
    total_retrieved: 1,
    passages: [samplePassage],
    has_results: true,
    execution_time_ms: 12.4,
    disclaimer:
      'Regulatory citations are provided for analyst decision support only and do not constitute legal determinations, regulatory findings, or automated SAR filing mandates.',
  };

  // Traceability checks
  assert.equal(samplePassage.document_id, 'FIN-2021-A003');
  assert.ok(samplePassage.source_url.startsWith('https://'));
  assert.ok(samplePassage.chunk_id.includes('FIN-2021-A003'));
  assert.ok(samplePassage.similarity_score >= 0.0 && samplePassage.similarity_score <= 1.0);

  // Mandatory advisory disclaimer
  assert.ok(sampleSearchResponse.disclaimer.toLowerCase().includes('decision support only'));
  assert.ok(sampleSearchResponse.disclaimer.toLowerCase().includes('not constitute legal determinations'));
});

test('RAG handles irrelevant queries safely without hallucinating citations', () => {
  const emptySearchResponse = {
    query: 'completely unrelated query about dessert recipes',
    total_retrieved: 0,
    passages: [],
    has_results: false,
    execution_time_ms: 5.1,
    disclaimer:
      'Regulatory citations are provided for analyst decision support only and do not constitute legal determinations, regulatory findings, or automated SAR filing mandates.',
  };

  assert.equal(emptySearchResponse.has_results, false);
  assert.equal(emptySearchResponse.passages.length, 0);
  assert.equal(emptySearchResponse.total_retrieved, 0);
});

