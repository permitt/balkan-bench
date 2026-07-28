import { render, screen } from '@testing-library/react'
import ScoreCell from './ScoreCell.jsx'

test('renders score, stdev, and proportional bar', () => {
  render(<ScoreCell cell={{ mean: 0.921, stdev: 0.012 }} active />)
  expect(screen.getByText('92.10')).toBeInTheDocument()
  expect(screen.getByText('± 1.20')).toBeInTheDocument()
  expect(screen.getByTestId('score-bar-fill')).toHaveStyle({ width: '92.1%' })
})

test('null cell renders dash without a bar', () => {
  render(<ScoreCell cell={null} active={false} />)
  expect(screen.getByText('-')).toBeInTheDocument()
  expect(screen.queryByTestId('score-bar-fill')).not.toBeInTheDocument()
})

test('active toggles class', () => {
  const { container } = render(<ScoreCell cell={{ mean: 0.5, stdev: 0 }} active />)
  expect(container.firstChild).toHaveClass('active')
})
