import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import LeaderboardTable from './LeaderboardTable.jsx'
import { boardFixture } from '../test/helpers.js'

function setup(props = {}) {
  const onRankBy = vi.fn()
  const onSelectModel = vi.fn()
  render(
    <LeaderboardTable
      data={boardFixture()}
      rankBy="avg"
      onRankBy={onRankBy}
      onSelectModel={onSelectModel}
      {...props}
    />,
  )
  return { onRankBy, onSelectModel }
}

test('renders rows sorted by avg with rank, params, hf link', () => {
  setup()
  const rows = screen.getAllByRole('row').slice(1) // skip header
  expect(within(rows[0]).getByText('model-a')).toBeInTheDocument()
  expect(within(rows[1]).getByText('model-b')).toBeInTheDocument()
  expect(within(rows[0]).getByRole('link', { name: 'org/model-a' }))
    .toHaveAttribute('href', 'https://huggingface.co/org/model-a')
  expect(within(rows[0]).getByText('110M')).toBeInTheDocument()
})

test('sorting by task reorders and header click fires onRankBy', async () => {
  const user = userEvent.setup()
  const { onRankBy } = setup({ rankBy: 'copa' })
  const rows = screen.getAllByRole('row').slice(1)
  expect(within(rows[0]).getByText('model-a')).toBeInTheDocument() // 0.88 copa
  await user.click(screen.getByRole('button', { name: /BoolQ/ }))
  expect(onRankBy).toHaveBeenCalledWith('boolq')
})

test('keyboard Enter/Space on a sortable header fires onRankBy', async () => {
  const user = userEvent.setup()
  const { onRankBy } = setup({ rankBy: 'copa' })
  const boolqHeader = screen.getByRole('button', { name: /BoolQ/ })
  boolqHeader.focus()
  await user.keyboard('{Enter}')
  expect(onRankBy).toHaveBeenCalledWith('boolq')
  onRankBy.mockClear()
  await user.keyboard(' ')
  expect(onRankBy).toHaveBeenCalledWith('boolq')
})

test('aria-sort marks only the currently ranked column', () => {
  setup({ rankBy: 'copa' })
  const activeHeader = screen.getByRole('columnheader', { name: /COPA/ })
  expect(activeHeader).toHaveAttribute('aria-sort', 'descending')
  expect(activeHeader.tagName).toBe('TH')
  expect(screen.getByRole('columnheader', { name: /BoolQ/ })).not.toHaveAttribute('aria-sort')
  expect(screen.getByRole('columnheader', { name: /Avg/ })).not.toHaveAttribute('aria-sort')
  // the nested button still supports click-to-sort
  expect(screen.getByRole('button', { name: /COPA/ })).toBeInTheDocument()
})

test('row click and Enter select the model', async () => {
  const user = userEvent.setup()
  const { onSelectModel } = setup()
  await user.click(screen.getByText('model-b'))
  expect(onSelectModel).toHaveBeenCalledWith(expect.objectContaining({ model: 'model-b' }))
  onSelectModel.mockClear()
  screen.getAllByRole('row').slice(1)[0].focus()
  await user.keyboard('{Enter}')
  expect(onSelectModel).toHaveBeenCalledWith(expect.objectContaining({ model: 'model-a' }))
})

test('empty board shows quiet message', () => {
  render(
    <LeaderboardTable data={boardFixture({ rows: [] })} rankBy="avg" onRankBy={() => {}} onSelectModel={() => {}} />,
  )
  expect(screen.getByText(/no results published yet/i)).toBeInTheDocument()
})

test('meta footer shows version, seeds, sponsor', () => {
  setup()
  expect(screen.getByText(/1\.0\.0/)).toBeInTheDocument()
  expect(screen.getByText(/5 seeds/)).toBeInTheDocument()
  expect(screen.getByText(/Recrewty/)).toBeInTheDocument()
})

test('avg column sits right after params', () => {
  setup()
  const headers = screen.getAllByRole('row')[0].cells
  const labels = Array.from(headers).map(h => h.textContent)
  expect(labels[2]).toBe('Params')
  expect(labels[3]).toBe('Avg')
})

test('avg cell sits right after params cell in body rows', () => {
  setup()
  const row = screen.getAllByRole('row')[1]
  expect(row.cells[2].textContent).toBe('110M')
  expect(within(row.cells[3]).getByText('90.00')).toBeInTheDocument()
})

test('malformed generated_at does not crash the table and omits the Generated span', () => {
  const data = boardFixture({ generated_at: 'not-a-date' })
  setup({ data })
  expect(screen.getByText('model-a')).toBeInTheDocument()
  expect(screen.queryByText(/generated/i)).not.toBeInTheDocument()
})

test('sle- display prefix is stripped from model names', () => {
  const data = boardFixture()
  data.rows[0] = { ...data.rows[0], model: 'sle-model-a' }
  setup({ data })
  expect(screen.getByText('model-a')).toBeInTheDocument()
  expect(screen.queryByText('sle-model-a')).not.toBeInTheDocument()
  expect(screen.getAllByRole('row')[1]).toHaveAccessibleName('model-a')
})
