import { vi } from 'vitest'

// mockBoards({ 'superglue-sr': payload, 'sle-sr': payload })
// URLs not in the map resolve as HTTP 404 (ok: false).
export function mockBoards(map) {
  vi.stubGlobal('fetch', vi.fn(async (url) => {
    const m = String(url).match(/\/leaderboards\/(.+)\/benchmark_results\.json$/)
    const payload = m && map[m[1]]
    if (!payload) return { ok: false, status: 404, json: async () => ({}) }
    return { ok: true, status: 200, json: async () => payload }
  }))
}

export function boardFixture(overrides = {}) {
  return {
    benchmark: 'superglue',
    language: 'sr',
    benchmark_version: '1.0.0',
    generated_at: '2026-07-28T00:00:00Z',
    sponsor: 'Recrewty',
    seeds: 5,
    ranked_tasks: ['boolq', 'copa'],
    task_primary_metrics: { boolq: 'acc', copa: 'acc' },
    rows: [
      {
        rank: 1, model: 'model-a', model_id: 'org/model-a', params: 110000000,
        params_display: '110M', complete: true, avg: 0.9,
        results: { boolq: { mean: 0.92, stdev: 0.01 }, copa: { mean: 0.88, stdev: 0.02 } },
      },
      {
        rank: 2, model: 'model-b', model_id: 'org/model-b', params: 110000000,
        params_display: '110M', complete: true, avg: 0.8,
        results: { boolq: { mean: 0.78, stdev: 0.01 }, copa: { mean: 0.82, stdev: 0.02 } },
      },
    ],
    ...overrides,
  }
}
