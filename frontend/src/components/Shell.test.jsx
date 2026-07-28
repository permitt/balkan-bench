import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter, RouterProvider, Navigate } from 'react-router-dom'
import Shell from './Shell.jsx'

function renderShell(initialPath = '/') {
  const router = createMemoryRouter(
    [{
      element: <Shell />,
      children: [
        { path: '/', element: <div>home-page</div> },
        { path: '/leaderboard', element: <div>lb-page</div> },
        { path: '/about', element: <div>about-page</div> },
        { path: '/submit', element: <div>submit-page</div> },
        { path: '/acknowledgements', element: <Navigate to="/about#acknowledgements" replace /> },
      ],
    }],
    { initialEntries: [initialPath] },
  )
  render(<RouterProvider router={router} />)
  return router
}

afterEach(() => {
  localStorage.clear()
  delete document.documentElement.dataset.theme
})

test('renders nav links and outlet', () => {
  renderShell('/')
  expect(screen.getByRole('link', { name: /leaderboard/i })).toHaveAttribute('href', '/leaderboard')
  expect(screen.getByRole('link', { name: /about/i })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /submit/i })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /github/i })).toHaveAttribute('href', expect.stringContaining('github.com'))
  expect(screen.getByText('home-page')).toBeInTheDocument()
})

test('theme toggle cycles auto -> light -> dark', async () => {
  const user = userEvent.setup()
  renderShell('/')
  const btn = screen.getByRole('button', { name: /theme: auto/i })
  await user.click(btn)
  expect(document.documentElement.dataset.theme).toBe('light')
  await user.click(screen.getByRole('button', { name: /theme: light/i }))
  expect(document.documentElement.dataset.theme).toBe('dark')
  await user.click(screen.getByRole('button', { name: /theme: dark/i }))
  expect(document.documentElement.dataset.theme).toBeUndefined()
})

test('acknowledgements redirects to about', () => {
  const router = renderShell('/acknowledgements')
  expect(router.state.location.pathname).toBe('/about')
  expect(router.state.location.hash).toBe('#acknowledgements')
})

test('footer credits sponsor', () => {
  renderShell('/')
  expect(screen.getByText(/compute sponsored by/i)).toBeInTheDocument()
})
