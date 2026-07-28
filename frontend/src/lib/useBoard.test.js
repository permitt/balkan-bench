import { renderHook, waitFor, act } from '@testing-library/react'
import { vi } from 'vitest'
import { useBoard } from './useBoard.js'
import { mockBoards, boardFixture } from '../test/helpers.js'

afterEach(() => vi.unstubAllGlobals())

const SG = { benchmark: 'superglue', language: 'sr', path: 'superglue-sr', available: true }
const SLE = { benchmark: 'sle', language: 'sr', path: 'sle-sr', available: true }

test('loads board data', async () => {
  mockBoards({ 'superglue-sr': boardFixture() })
  const { result } = renderHook(() => useBoard(SG))
  expect(result.current.data).toBeNull()
  await waitFor(() => expect(result.current.data).not.toBeNull())
  expect(result.current.data.rows).toHaveLength(2)
  expect(result.current.apiData).toBeNull()
  expect(result.current.error).toBeNull()
})

test('sets error on failed fetch and retry refetches', async () => {
  mockBoards({})
  const { result } = renderHook(() => useBoard(SG))
  await waitFor(() => expect(result.current.error).toBe('HTTP 404'))
  mockBoards({ 'superglue-sr': boardFixture() })
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.data).not.toBeNull())
  expect(result.current.error).toBeNull()
})

test('sle fetches api board too; missing api export means empty rows', async () => {
  mockBoards({ 'sle-sr': boardFixture({ benchmark: 'sle' }) })
  const { result } = renderHook(() => useBoard(SLE))
  await waitFor(() => expect(result.current.data).not.toBeNull())
  await waitFor(() => expect(result.current.apiData).toEqual({ rows: [] }))
})

test('refetches when target changes', async () => {
  mockBoards({ 'superglue-sr': boardFixture(), 'superglue-hr': boardFixture({ language: 'hr' }) })
  const { result, rerender } = renderHook(({ t }) => useBoard(t), { initialProps: { t: SG } })
  await waitFor(() => expect(result.current.data?.language).toBe('sr'))
  rerender({ t: { ...SG, language: 'hr', path: 'superglue-hr' } })
  await waitFor(() => expect(result.current.data?.language).toBe('hr'))
})
