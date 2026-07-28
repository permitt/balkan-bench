import { renderHook, act } from '@testing-library/react'
import { useTheme } from './useTheme.js'

afterEach(() => {
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

test('defaults to auto with no attribute', () => {
  const { result } = renderHook(() => useTheme())
  expect(result.current.theme).toBe('auto')
  expect(document.documentElement.dataset.theme).toBeUndefined()
})

test('setTheme(dark) sets attribute and persists', () => {
  const { result } = renderHook(() => useTheme())
  act(() => result.current.setTheme('dark'))
  expect(document.documentElement.dataset.theme).toBe('dark')
  expect(localStorage.getItem('bb-theme')).toBe('dark')
})

test('reads persisted theme on mount', () => {
  localStorage.setItem('bb-theme', 'light')
  const { result } = renderHook(() => useTheme())
  expect(result.current.theme).toBe('light')
  expect(document.documentElement.dataset.theme).toBe('light')
})

test('setTheme(auto) clears storage and attribute', () => {
  localStorage.setItem('bb-theme', 'dark')
  const { result } = renderHook(() => useTheme())
  act(() => result.current.setTheme('auto'))
  expect(localStorage.getItem('bb-theme')).toBeNull()
  expect(document.documentElement.dataset.theme).toBeUndefined()
})
