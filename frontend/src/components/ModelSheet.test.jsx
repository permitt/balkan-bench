import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import ModelSheet from './ModelSheet.jsx'
import { boardFixture } from '../test/helpers.js'

const board = boardFixture()
const row = board.rows[0]

test('renders nothing when row is null', () => {
  render(<ModelSheet row={null} board={board} protocol={null} onClose={() => {}} />)
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

test('renders dialog with model info and per-task scores', () => {
  render(<ModelSheet row={row} board={board} protocol="Open weights" onClose={() => {}} />)
  const dialog = screen.getByRole('dialog', { name: 'model-a' })
  expect(dialog).toHaveAttribute('aria-modal', 'true')
  expect(screen.getByRole('link', { name: 'org/model-a' }))
    .toHaveAttribute('href', 'https://huggingface.co/org/model-a')
  expect(screen.getByText('110M')).toBeInTheDocument()
  expect(screen.getByText('BoolQ')).toBeInTheDocument()
  expect(screen.getByText(/boolean questions/i)).toBeInTheDocument()
  expect(screen.getByText('Open weights')).toBeInTheDocument()
})

test('focus moves to dialog on open', () => {
  render(<ModelSheet row={row} board={board} protocol={null} onClose={() => {}} />)
  expect(screen.getByRole('dialog')).toHaveFocus()
})

test('esc, scrim, and close button all close', async () => {
  const user = userEvent.setup()
  const onClose = vi.fn()
  render(<ModelSheet row={row} board={board} protocol={null} onClose={onClose} />)
  await user.keyboard('{Escape}')
  await user.click(screen.getByTestId('sheet-scrim'))
  await user.click(screen.getByRole('button', { name: /close/i }))
  expect(onClose).toHaveBeenCalledTimes(3)
})
