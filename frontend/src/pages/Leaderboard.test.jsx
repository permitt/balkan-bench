import { render, screen, waitFor, within } from '@testing-library/react'
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
  await user.click(screen.getByRole('button', { name: /BoolQ/ }))
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

test('closed-API board keeps its own rows and metric labels', async () => {
  const user = userEvent.setup()
  renderPage('/leaderboard?benchmark=sle&model=api-model-x', {
    'sle-sr': boardFixture({
      benchmark: 'sle',
      ranked_tasks: ['boolq', 'copa'],
      task_primary_metrics: { boolq: 'acc_norm', copa: 'acc_norm' },
    }),
    'sle-sr-api': boardFixture({
      benchmark: 'sle',
      seeds: undefined,
      ranked_tasks: ['boolq', 'copa'],
      task_primary_metrics: { boolq: 'acc', copa: 'acc' },
      rows: [
        {
          rank: 1, model: 'api-model-x', model_id: 'org/api-model-x', params: 0,
          params_display: 'closed', complete: true, avg: 0.7,
          results: { boolq: { mean: 0.71, stdev: 0.01 }, copa: { mean: 0.69, stdev: 0.02 } },
        },
      ],
    }),
  })

  const dialog = await screen.findByRole('dialog', { name: 'api-model-x' })
  expect(dialog).toBeInTheDocument()

  await screen.findByRole('heading', { name: /closed api - generative protocol/i })
  const table = screen.getAllByRole('table')[1] // second board = closed API
  expect(within(table).getByText('api-model-x')).toBeInTheDocument()
  // closed board's own metric label ('acc') must render, not the open
  // board's 'acc_norm' labels forced onto it by the old spread-merge bug.
  expect(within(table).getAllByText('acc').length).toBeGreaterThan(0)
  expect(within(table).queryByText('acc_norm')).not.toBeInTheDocument()

  await user.click(screen.getAllByRole('button', { name: /close/i })[0])
})

test('fetch failure shows error card, retry recovers', async () => {
  const user = userEvent.setup()
  renderPage('/leaderboard', {})
  expect(await screen.findByText(/failed to load/i)).toBeInTheDocument()
  mockBoards({ 'superglue-sr': boardFixture() })
  await user.click(screen.getByRole('button', { name: /retry/i }))
  expect(await screen.findByText('model-a')).toBeInTheDocument()
})
