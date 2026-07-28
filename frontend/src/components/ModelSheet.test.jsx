import { render, screen, fireEvent } from '@testing-library/react'
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

test('inerts the page behind the sheet and traps Tab focus within the panel', () => {
  render(
    <>
      <nav className="shell-nav"><a href="/x">nav link</a></nav>
      <main className="shell-main"><button type="button">page btn</button></main>
      <ModelSheet row={row} board={board} protocol={null} onClose={() => {}} />
    </>,
  )

  expect(document.querySelector('.shell-main')).toHaveAttribute('inert')
  expect(document.querySelector('.shell-nav')).toHaveAttribute('inert')

  const dialog = screen.getByRole('dialog')
  const closeBtn = screen.getByRole('button', { name: /close/i })
  const hfLink = screen.getByRole('link', { name: 'org/model-a' })

  // Tab from the last focusable wraps to the first.
  closeBtn.focus()
  fireEvent.keyDown(dialog, { key: 'Tab' })
  expect(hfLink).toHaveFocus()

  // Shift+Tab from the first focusable wraps to the last.
  hfLink.focus()
  fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true })
  expect(closeBtn).toHaveFocus()
})
