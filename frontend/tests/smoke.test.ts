/**
 * Smoke test verifying that basic components and types are exportable and valid.
 */
import { checkHealth } from '../src/services/api';

describe('Frontend API Service Smoke Test', () => {
  it('should export checkHealth function', () => {
    expect(typeof checkHealth).toBe('function');
  });
});

