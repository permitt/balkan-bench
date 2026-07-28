import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import Segmented from './Segmented.jsx'

const options = [
  { value: 'sr', label: 'SR', sublabel: 'Srpski' },
  { value: 'hr', label: 'HR', sublabel: 'Hrvatski' },
  { value: 'bs', label: 'BS', sublabel: 'Bosanski', disabled: true, badge: 'v1.1' },
]

test('renders radiogroup with checked state', () => {
  render(<Segmented label="Language" value="sr" onChange={() => {}} options={options} />)
  expect(screen.getByRole('radiogroup', { name: 'Language' })).toBeInTheDocument()
  expect(screen.getByRole('radio', { name: /SR/ })).toHaveAttribute('aria-checked', 'true')
  expect(screen.getByRole('radio', { name: /HR/ })).toHaveAttribute('aria-checked', 'false')
})

test('clicking an option fires onChange', async () => {
  const user = userEvent.setup()
  const onChange = vi.fn()
  render(<Segmented label="Language" value="sr" onChange={onChange} options={options} />)
  await user.click(screen.getByRole('radio', { name: /HR/ }))
  expect(onChange).toHaveBeenCalledWith('hr')
})

test('disabled option shows badge and does not fire', async () => {
  const user = userEvent.setup()
  const onChange = vi.fn()
  render(<Segmented label="Language" value="sr" onChange={onChange} options={options} />)
  const bs = screen.getByRole('radio', { name: /BS/ })
  expect(bs).toBeDisabled()
  expect(screen.getByText('v1.1')).toBeInTheDocument()
  await user.click(bs)
  expect(onChange).not.toHaveBeenCalled()
})

test('ArrowRight/ArrowDown select the next option', () => {
  const onChange = vi.fn()
  render(<Segmented label="Language" value="sr" onChange={onChange} options={options} />)
  const group = screen.getByRole('radiogroup', { name: 'Language' })
  fireEvent.keyDown(group, { key: 'ArrowRight' })
  expect(onChange).toHaveBeenCalledWith('hr')
  onChange.mockClear()
  fireEvent.keyDown(group, { key: 'ArrowDown' })
  expect(onChange).toHaveBeenCalledWith('hr')
})

test('ArrowLeft/ArrowUp select the previous option, skipping disabled and wrapping', () => {
  const onChange = vi.fn()
  render(<Segmented label="Language" value="sr" onChange={onChange} options={options} />)
  const group = screen.getByRole('radiogroup', { name: 'Language' })
  // from sr (idx 0), previous wraps past the disabled bs option to hr
  fireEvent.keyDown(group, { key: 'ArrowLeft' })
  expect(onChange).toHaveBeenCalledWith('hr')
  onChange.mockClear()
  fireEvent.keyDown(group, { key: 'ArrowUp' })
  expect(onChange).toHaveBeenCalledWith('hr')
})

test('ArrowRight from the last enabled option wraps around disabled options to the first', () => {
  const onChange = vi.fn()
  render(<Segmented label="Language" value="hr" onChange={onChange} options={options} />)
  const group = screen.getByRole('radiogroup', { name: 'Language' })
  fireEvent.keyDown(group, { key: 'ArrowRight' })
  expect(onChange).toHaveBeenCalledWith('sr')
})
