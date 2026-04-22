import { useEffect, useMemo, useState } from 'react'
import Topbar from '../components/Topbar.jsx'
import Nav from '../components/Nav.jsx'
import Footer from '../components/Footer.jsx'
import '../styles/leaderboard.css'

const TASK_LABELS = {
  boolq: 'BoolQ',
  cb: 'CB',
  copa: 'COPA',
  rte: 'RTE',
  multirc: 'MultiRC',
  wsc: 'WSC',
}

function formatCell(cell) {
  if (cell === null || cell === undefined) return { main: '—', stdev: null }
  const mean = Number(cell.mean).toFixed(2)
  const stdev = cell.stdev === undefined ? null : Number(cell.stdev).toFixed(2)
  return { main: mean, stdev }
}

export default function Leaderboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [sortKey, setSortKey] = useState('avg')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    fetch('/leaderboards/superglue-serbian/benchmark_results.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  const sortedRows = useMemo(() => {
    if (!data) return []
    const rows = [...data.rows]
    rows.sort((a, b) => {
      const av = sortValue(a, sortKey)
      const bv = sortValue(b, sortKey)
      if (av === null && bv === null) return 0
      if (av === null) return 1
      if (bv === null) return -1
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return rows
  }, [data, sortKey, sortDir])

  const clickSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  return (
    <>
      <Topbar />
      <Nav />
      <section className="lb-wrap">
        <div className="lb-head">
          <div className="lb-eyebrow">
            <span className="chip">LEADERBOARD</span>
            <span>SUPERGLUE · SR · V0.1</span>
          </div>
          <h1 className="lb-title">
            Every model, <span className="stroke">measured</span><span className="slash">.</span>
          </h1>
          <p className="lb-sub">
            Serbian SuperGLUE, 6 ranked tasks, 5 seeds each, primary metric per task.
            {' '}Compute sponsored by <b>Recrewty</b>.
          </p>
        </div>

        {error && <div className="lb-error">Failed to load leaderboard: {error}</div>}

        {data && (
          <div className="lb-table-wrap">
            <table className="lb-table">
              <thead>
                <tr>
                  <th className="lb-rank">#</th>
                  <th className="lb-model">Model</th>
                  <th className="lb-params">Params</th>
                  {data.ranked_tasks.map((t) => (
                    <th key={t} className="lb-num" onClick={() => clickSort(t)}>
                      {TASK_LABELS[t] || t}
                      <span className="lb-metric">({data.task_primary_metrics[t]})</span>
                      <span className="lb-caret">{sortKey === t ? (sortDir === 'asc' ? '▲' : '▼') : ''}</span>
                    </th>
                  ))}
                  <th className="lb-num lb-avg" onClick={() => clickSort('avg')}>
                    Avg
                    <span className="lb-caret">{sortKey === 'avg' ? (sortDir === 'asc' ? '▲' : '▼') : ''}</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row) => (
                  <tr key={row.model} className={!row.complete ? 'lb-partial' : ''}>
                    <td className="lb-rank">{row.rank ?? (row.partial_flag || '—')}</td>
                    <td className="lb-model">
                      <div className="lb-model-name">{row.model}</div>
                      <div className="lb-model-id">{row.model_id}</div>
                    </td>
                    <td className="lb-params">{row.params_display}</td>
                    {data.ranked_tasks.map((t) => {
                      const { main, stdev } = formatCell(row.results[t])
                      return (
                        <td key={t} className="lb-num">
                          <div className="lb-cell-main">{main}</div>
                          {stdev !== null && <div className="lb-cell-stdev">± {stdev}</div>}
                        </td>
                      )
                    })}
                    <td className="lb-num lb-avg">
                      <div className="lb-cell-main"><b>{row.avg.toFixed(2)}</b></div>
                      {!row.complete && <div className="lb-cell-stdev">{row.partial_flag}</div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="lb-meta">
              <span>Benchmark version <b>{data.benchmark_version}</b></span>
              <span className="sep">/</span>
              <span>{data.seeds} seeds</span>
              <span className="sep">/</span>
              <span>Generated {new Date(data.generated_at).toISOString().slice(0, 10)}</span>
              <span className="sep">/</span>
              <span>Compute sponsored by <b>{data.sponsor}</b></span>
            </div>
          </div>
        )}

        {!data && !error && <div className="lb-loading">Loading leaderboard…</div>}
      </section>
      <Footer />
    </>
  )
}

function sortValue(row, key) {
  if (key === 'avg') return row.avg ?? null
  const cell = row.results[key]
  return cell ? cell.mean : null
}
