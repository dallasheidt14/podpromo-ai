import { isTerminalProgress } from '../shared/api';

describe('isTerminalProgress', () => {
  it('treats completed as terminal', () => {
    expect(isTerminalProgress({ stage: 'completed', percent: 100 })).toBe(true);
  });
  it('treats error as terminal', () => {
    expect(isTerminalProgress({ stage: 'error', percent: 0 })).toBe(true);
  });
  it('treats scoring@100 as terminal', () => {
    expect(isTerminalProgress({ stage: 'scoring', percent: 100 })).toBe(true);
  });
  it('non-terminal states are false', () => {
    expect(isTerminalProgress({ stage: 'scoring', percent: 95 })).toBe(false);
  });
});
