import { render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import About from './About.jsx'

function renderAbout() {
  const router = createMemoryRouter(
    [{ path: '/about', element: <About /> }],
    { initialEntries: ['/about'] },
  )
  render(<RouterProvider router={router} />)
}

test('renders three anchored sections with in-page nav', () => {
  renderAbout()
  expect(document.getElementById('benchmarks')).toBeInTheDocument()
  expect(document.getElementById('methodology')).toBeInTheDocument()
  expect(document.getElementById('acknowledgements')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /acknowledgements/i })).toHaveAttribute('href', '/about#acknowledgements')
})

test('carries over key content', () => {
  renderAbout()
  expect(screen.getByText(/67,313 items/)).toBeInTheDocument()
  expect(screen.getByText(/Daria Milošević/)).toBeInTheDocument()
  expect(screen.getByText(/hidden test labels/i)).toBeInTheDocument()
})
