import { render, screen, waitForElementToBeRemoved } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import RankMenu from './RankMenu.jsx'

const tasks = ['boolq', 'copa']
const metrics = { boolq: 'acc', copa: 'acc' }

test('pill shows current selection', () => {
  render(<RankMenu value="avg" onChange={() => {}} tasks={tasks} metrics={metrics} />)
  expect(screen.getByRole('button', { name: /rank by: avg/i })).toHaveAttribute('aria-expanded', 'false')
})

test('opens menu, selects a task, closes', async () => {
  const user = userEvent.setup()
  const onChange = vi.fn()
  render(<RankMenu value="avg" onChange={onChange} tasks={tasks} metrics={metrics} />)
  await user.click(screen.getByRole('button', { name: /rank by/i }))
  expect(screen.getByRole('menu')).toBeInTheDocument()
  const item = screen.getByRole('menuitemradio', { name: /BoolQ/ })
  await user.click(item)
  expect(onChange).toHaveBeenCalledWith('boolq')
  await waitForElementToBeRemoved(() => screen.queryByRole('menu'))
})

test('marks current selection checked and shows descriptions', async () => {
  const user = userEvent.setup()
  render(<RankMenu value="boolq" onChange={() => {}} tasks={tasks} metrics={metrics} />)
  await user.click(screen.getByRole('button', { name: /rank by: boolq/i }))
  expect(screen.getByRole('menuitemradio', { name: /BoolQ/ })).toHaveAttribute('aria-checked', 'true')
  expect(screen.getByText(/boolean questions over a short passage/i)).toBeInTheDocument()
})

test('escape closes the menu', async () => {
  const user = userEvent.setup()
  render(<RankMenu value="avg" onChange={() => {}} tasks={tasks} metrics={metrics} />)
  await user.click(screen.getByRole('button', { name: /rank by/i }))
  await user.keyboard('{Escape}')
  await waitForElementToBeRemoved(() => screen.queryByRole('menu'))
})
