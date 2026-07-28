import { render, screen, within } from '@testing-library/react'
import { vi } from 'vitest'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import Home from './Home.jsx'
import { mockBoards, boardFixture } from '../test/helpers.js'

afterEach(() => vi.unstubAllGlobals())

function renderHome() {
  mockBoards({ 'superglue-sr': boardFixture() })
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
  expect(screen.getByRole('link', { name: /full leaderboard/i })).toHaveAttribute('href', '/leaderboard')
})

test('facts strip shows items, languages, model count', async () => {
  renderHome()
  await screen.findByText('model-a')
  const facts = within(screen.getByRole('region', { name: /key figures/i }))
  expect(facts.getByText('67,313')).toBeInTheDocument()
  expect(facts.getByText('2')).toBeInTheDocument() // models in fixture
})

test('benchmark roster lists all four tracks with status', () => {
  renderHome()
  expect(screen.getByText('SuperGLUE')).toBeInTheDocument()
  expect(screen.getByText('Serbian-LLM-Eval')).toBeInTheDocument()
  expect(screen.getByText('MTEB-BCMS')).toBeInTheDocument()
  expect(screen.getByText('LLM Arena')).toBeInTheDocument()
  expect(screen.getAllByText(/planned/i).length).toBe(2)
})
