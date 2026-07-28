import { render, screen, within } from '@testing-library/react'
import { vi } from 'vitest'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import Home from './Home.jsx'
import { mockBoards, boardFixture } from '../test/helpers.js'

afterEach(() => vi.unstubAllGlobals())

function renderHome() {
  mockBoards({
    'superglue-sr': boardFixture(),
    'sle-sr': boardFixture({
      benchmark: 'sle',
      seeds: undefined,
      rows: [
        {
          rank: 1, model: 'sle-yugo-model', model_id: 'org/yugo-model', params: 7000000000,
          params_display: '7B', complete: true, avg: 0.62,
          results: { boolq: { mean: 0.7 }, copa: { mean: 0.54 } },
        },
      ],
    }),
  })
  const router = createMemoryRouter(
    [{ path: '/', element: <Home /> }],
    { initialEntries: ['/'] },
  )
  render(<RouterProvider router={router} />)
}

test('hero renders headline and CTAs', () => {
  renderHome()
  expect(screen.getByRole('heading', { level: 1, name: /every model, measured/i })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /view leaderboard/i })).toHaveAttribute('href', '/leaderboard')
  expect(screen.getByRole('link', { name: /read methodology/i })).toHaveAttribute('href', '/about')
})

test('live preview shows top rows linking to leaderboard', async () => {
  renderHome()
  expect(await screen.findByText('model-a')).toBeInTheDocument()
  const superglueCard = screen.getByRole('region', { name: /superglue top models/i })
  expect(within(superglueCard).getByRole('link', { name: /full leaderboard/i }))
    .toHaveAttribute('href', '/leaderboard')
})

test('sle preview card comes before superglue and links to the sle board', async () => {
  renderHome()
  const sleCard = screen.getByRole('region', { name: /serbian-llm-eval top models/i })
  expect(within(sleCard).getByRole('link', { name: /full leaderboard/i }))
    .toHaveAttribute('href', '/leaderboard?benchmark=sle')
  expect(await within(sleCard).findByText('yugo-model')).toBeInTheDocument() // sle- prefix stripped
  const superglueCard = screen.getByRole('region', { name: /superglue top models/i })
  expect(sleCard.compareDocumentPosition(superglueCard) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
})

test('facts strip shows items, models, languages across both suites', async () => {
  renderHome()
  await screen.findByText('model-a')
  const facts = within(screen.getByRole('region', { name: /key figures/i }))
  expect(facts.getByText('109,332')).toBeInTheDocument()
  expect(facts.getByText('19')).toBeInTheDocument()
  expect(facts.getByText('3')).toBeInTheDocument()
})

test('vision section explains the mission with traceability claim', () => {
  renderHome()
  expect(screen.getByRole('heading', { name: /why this exists/i })).toBeInTheDocument()
  expect(screen.getByText(/traceable to the exact dataset, config, and seed/i)).toBeInTheDocument()
})

test('pillars cover open, reproducible, community with submit link', () => {
  renderHome()
  expect(screen.getByRole('heading', { name: /open by default/i })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: /reproducible to the seed/i })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: /built by the region/i })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /add your model/i })).toHaveAttribute('href', '/submit')
})

test('benchmark roster lists all four tracks with status', () => {
  renderHome()
  expect(screen.getByText('SuperGLUE')).toBeInTheDocument()
  expect(screen.getByText('Serbian-LLM-Eval')).toBeInTheDocument()
  expect(screen.getByText('MTEB-BCMS')).toBeInTheDocument()
  expect(screen.getByText('LLM Arena')).toBeInTheDocument()
  expect(screen.getAllByText(/planned/i).length).toBe(2)
})
