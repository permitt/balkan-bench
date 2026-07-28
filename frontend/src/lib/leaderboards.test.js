import { describe, test, expect } from 'vitest'
import {
  formatCell, formatAvg, sortRows, sortValue,
  resolveBoard, boardEntriesFor, boardUrl, LEADERBOARDS, FACTS,
} from './leaderboards.js'

describe('leaderboards', () => {
  test('formatCell rescales 0-1 to 0-100 with stdev', () => {
    expect(formatCell({ mean: 0.921, stdev: 0.012 })).toEqual({ main: '92.10', stdev: '1.20' })
  })

  test('formatCell handles missing cell and missing stdev', () => {
    expect(formatCell(null)).toEqual({ main: '-', stdev: null })
    expect(formatCell({ mean: 0.5 })).toEqual({ main: '50.00', stdev: null })
  })

  test('formatAvg renders row average on 0-100 scale', () => {
    expect(formatAvg({ avg: 0.8765 })).toBe('87.65')
  })

  test('sortRows by task puts nulls last, descending means first', () => {
    const rows = [
      { model: 'a', results: { boolq: { mean: 0.5 } }, avg: 0.5 },
      { model: 'b', results: {}, avg: 0.9 },
      { model: 'c', results: { boolq: { mean: 0.9 } }, avg: 0.7 },
    ]
    expect(sortRows(rows, 'boolq').map(r => r.model)).toEqual(['c', 'a', 'b'])
    expect(rows[0].model).toBe('a') // non-mutating
  })

  test('sortValue reads avg or task mean', () => {
    const row = { avg: 0.7, results: { boolq: { mean: 0.6 } } }
    expect(sortValue(row, 'avg')).toBe(0.7)
    expect(sortValue(row, 'boolq')).toBe(0.6)
    expect(sortValue(row, 'copa')).toBeNull()
  })

  test('resolveBoard falls back to first available board', () => {
    expect(resolveBoard('superglue', 'sr').path).toBe('superglue-sr')
    expect(resolveBoard('superglue', 'bs').path).toBe('superglue-sr') // bs unavailable
    expect(resolveBoard('nope', 'xx').path).toBe('superglue-sr')
  })

  test('resolveBoard falls back to the same benchmark in another language before the global default', () => {
    expect(resolveBoard('sle', 'hr').path).toBe('sle-sr')
  })

  test('boardEntriesFor filters by benchmark', () => {
    expect(boardEntriesFor('sle').every(e => e.benchmark === 'sle')).toBe(true)
    expect(boardEntriesFor('superglue').length).toBe(4)
  })

  test('boardUrl builds the fetch path', () => {
    expect(boardUrl('sle-sr')).toBe('/leaderboards/sle-sr/benchmark_results.json')
  })

  test('facts constants', () => {
    expect(FACTS.items).toBe(109332)
    expect(FACTS.languageCount).toBe(3)
    expect(FACTS.modelCount).toBe(19)
    expect(LEADERBOARDS.some(l => l.path === 'sle-sr' && l.available)).toBe(true)
  })
})
