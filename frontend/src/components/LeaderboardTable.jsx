import { useMemo } from 'react'
import { LayoutGroup, motion } from 'motion/react' // eslint-disable-line no-unused-vars
import { sortRows, sortValue, displayModelName, TASK_LABELS } from '../lib/leaderboards.js'
import ScoreCell from './ScoreCell.jsx'
import '../styles/leaderboard.css'

const ROW_SPRING = { type: 'spring', bounce: 0, duration: 0.4 }

export default function LeaderboardTable({ data, rankBy, onRankBy, onSelectModel }) {
  const sorted = useMemo(() => sortRows(data.rows, rankBy), [data, rankBy])

  if (data.rows.length === 0) {
    return <div className="lb-empty">No results published yet.</div>
  }

  return (
    <div className="lb-table-wrap">
      <table className="lb-table">
        <thead>
          <tr>
            <th className="lb-rank" scope="col">#</th>
            <th className="lb-model" scope="col">Model</th>
            <th scope="col">Params</th>
            <th
              scope="col"
              className={`lb-num lb-avg lb-sortable ${rankBy === 'avg' ? 'lb-col-active' : ''}`}
              aria-sort={rankBy === 'avg' ? 'descending' : undefined}
            >
              <button type="button" className="lb-sort-btn" onClick={() => onRankBy('avg')}>
                Avg
              </button>
            </th>
            {data.ranked_tasks.map(t => (
              <th
                key={t}
                scope="col"
                className={`lb-num lb-sortable ${rankBy === t ? 'lb-col-active' : ''}`}
                aria-sort={rankBy === t ? 'descending' : undefined}
              >
                <button type="button" className="lb-sort-btn" onClick={() => onRankBy(t)}>
                  {TASK_LABELS[t] || t}
                  <span className="lb-metric num">{data.task_primary_metrics[t]}</span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <LayoutGroup>
          <tbody>
            {sorted.map((row, i) => {
              const v = sortValue(row, rankBy)
              const displayRank = rankBy === 'avg'
                ? row.rank ?? (row.partial_flag || '-')
                : v === null ? '-' : i + 1
              return (
                <motion.tr
                  key={row.model}
                  layout="position"
                  transition={ROW_SPRING}
                  className={`lb-row ${i === 0 ? 'lb-first' : ''} ${!row.complete ? 'lb-partial' : ''}`}
                  tabIndex={0}
                  aria-label={displayModelName(row.model)}
                  onClick={() => onSelectModel(row)}
                  onKeyDown={(e) => { if (e.key === 'Enter') onSelectModel(row) }}
                >
                  <td className="lb-rank num">{displayRank}</td>
                  <td className="lb-model">
                    <div className="lb-model-name">{displayModelName(row.model)}</div>
                    <div className="lb-model-id">
                      {row.model_id && row.model_id.includes('/') ? (
                        <a
                          href={`https://huggingface.co/${row.model_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {row.model_id}
                        </a>
                      ) : row.model_id}
                    </div>
                  </td>
                  <td className="num">{row.params_display}</td>
                  <td className="lb-num lb-avg">
                    <ScoreCell cell={{ mean: row.avg }} active={rankBy === 'avg'} />
                    {!row.complete && <div className="lb-flag num">{row.partial_flag}</div>}
                  </td>
                  {data.ranked_tasks.map(t => (
                    <td key={t} className="lb-num">
                      <ScoreCell cell={row.results[t] ?? null} active={rankBy === t} />
                    </td>
                  ))}
                </motion.tr>
              )
            })}
          </tbody>
        </LayoutGroup>
      </table>
      <div className="lb-meta">
        <span>Benchmark version <b>{data.benchmark_version}</b></span>
        {data.seeds !== undefined && <span>{data.seeds} seeds</span>}
        <span>Generated {new Date(data.generated_at).toISOString().slice(0, 10)}</span>
        <span>Compute sponsored by <b>{data.sponsor}</b></span>
      </div>
    </div>
  )
}
