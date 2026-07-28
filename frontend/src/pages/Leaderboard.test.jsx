import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import Leaderboard from './Leaderboard.jsx'
import { mockBoards, boardFixture } from '../test/helpers.js'

afterEach(() => vi.unstubAllGlobals())

function renderPage(url = '/leaderboard', boards) {
  mockBoards(boards ?? {
    'superglue-sr': boardFixture(),
    'superglue-hr': boardFixture({ language: 'hr' }),
  })
  const router = createMemoryRouter(
    [{ path: '/leaderboard', element: <Leaderboard /> }],
    { initialEntries: [url] },
  )
  render(<RouterProvider router={router} />)
  return router
}

test('loads default board and renders table', async () => {
  renderPage()
  expect(screen.getByTestId('skeleton')).toBeInTheDocument()
  expect(await screen.findByText('model-a')).toBeInTheDocument()
})

test('language switch updates URL param', async () => {
  const user = userEvent.setup()
  const router = renderPage()
  await screen.findByText('model-a')
  await user.click(screen.getByRole('radio', { name: /HR/ }))
  await waitFor(() =>
    expect(new URLSearchParams(router.state.location.search).get('lang')).toBe('hr'))
})

test('rank-by column header updates task param', async () => {
  const user = userEvent.setup()
  const router = renderPage()
  await screen.findByText('model-a')
  await user.click(screen.getByRole('columnheader', { name: /BoolQ/ }))
  await waitFor(() =>
    expect(new URLSearchParams(router.state.location.search).get('task')).toBe('boolq'))
})

test('row click opens sheet and sets model param; close clears it', async () => {
  const user = userEvent.setup()
  const router = renderPage()
  await user.click(await screen.findByText('model-a'))
  expect(await screen.findByRole('dialog', { name: 'model-a' })).toBeInTheDocument()
  expect(new URLSearchParams(router.state.location.search).get('model')).toBe('model-a')
  await user.click(screen.getByRole('button', { name: /close/i }))
  await waitFor(() =>
    expect(new URLSearchParams(router.state.location.search).get('model')).toBeNull())
})

test('deep link with model param opens sheet', async () => {
  renderPage('/leaderboard?model=model-b')
  expect(await screen.findByRole('dialog', { name: 'model-b' })).toBeInTheDocument()
})

test('sle renders two boards', async () => {
  renderPage('/leaderboard?benchmark=sle', {
    'sle-sr': boardFixture({ benchmark: 'sle' }),
    'sle-sr-api': boardFixture({ benchmark: 'sle', seeds: undefined }),
  })
  expect(await screen.findByText(/open weights - loglikelihood protocol/i)).toBeInTheDocument()
  expect(screen.getByText(/closed api - generative protocol/i)).toBeInTheDocument()
  expect(screen.getByText(/not comparable/i)).toBeInTheDocument()
})

test('fetch failure shows error card, retry recovers', async () => {
  const user = userEvent.setup()
  renderPage('/leaderboard', {})
  expect(await screen.findByText(/failed to load/i)).toBeInTheDocument()
  mockBoards({ 'superglue-sr': boardFixture() })
  await user.click(screen.getByRole('button', { name: /retry/i }))
  expect(await screen.findByText('model-a')).toBeInTheDocument()
})
