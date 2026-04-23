import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
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

// Discoverable leaderboards. When a new (benchmark, language) pair publishes
// its benchmark_results.json, flip `available: true`; no other code changes.
const LEADERBOARDS = [
  { benchmark: 'superglue', language: 'sr',  path: 'superglue-serbian',    label: 'SuperGLUE · Serbian',      available: true  },
  { benchmark: 'superglue', language: 'hr',  path: 'superglue-croatian',   label: 'SuperGLUE · Croatian',     available: false },
  { benchmark: 'superglue', language: 'cnr', path: 'superglue-montenegrin',label: 'SuperGLUE · Montenegrin',  available: false },
  { benchmark: 'superglue', language: 'bs',  path: 'superglue-bosnian',    label: 'SuperGLUE · Bosnian',      available: false },
  { benchmark: 'sle',       language: 'sr',  path: 'sle-serbian',          label: 'Serbian-LLM-Eval · Serbian', available: false },
]

function formatCell(cell) {
  if (cell === null || cell === undefined) return { main: '—', stdev: null }
  const mean = Number(cell.mean).toFixed(2)
  const stdev = cell.stdev === undefined ? null : Number(cell.stdev).toFixed(2)
  return { main: mean, stdev }
}

function sortValue(row, rankBy) {
  if (rankBy === 'avg') return row.avg ?? null
  const cell = row.results[rankBy]
  return cell ? cell.mean : null
}

export default function Leaderboard() {
  const [params, setParams] = useSearchParams()
  const lang = params.get('lang') || 'sr'
  const rankBy = params.get('task') || 'avg'

  const target =
    LEADERBOARDS.find((l) => l.language === lang && l.available) ?? LEADERBOARDS[0]

  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setData(null)
    setError(null)
    fetch(`/leaderboards/${target.path}/benchmark_results.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setData)
      .catch((e) => setError(e.message))
  }, [target.path])

  const sortedRows = useMemo(() => {
    if (!data) return []
    const rows = [...data.rows]
    rows.sort((a, b) => {
      const av = sortValue(a, rankBy)
      const bv = sortValue(b, rankBy)
      if (av === null && bv === null) return 0
      if (av === null) return 1
      if (bv === null) return -1
      return bv - av
    })
    return rows
  }, [data, rankBy])

  const setLang = (nextLang) => {
    const next = new URLSearchParams(params)
    next.set('lang', nextLang)
    setParams(next, { replace: true })
  }

  const setRankBy = (nextTask) => {
    const next = new URLSearchParams(params)
    if (nextTask === 'avg') next.delete('task')
    else next.set('task', nextTask)
    setParams(next, { replace: true })
  }

  return (
    <>
      <Topbar />
      <Nav />
      <section className="lb-wrap">
        <div className="lb-head">
          <div className="lb-eyebrow">
            <span className="chip">LEADERBOARD</span>
            <span>{target.label.toUpperCase()} · V0.1</span>
          </div>
          <h1 className="lb-title">
            Every model, <span className="stroke">measured</span><span className="slash">.</span>
          </h1>
          <p className="lb-sub">
            {data ? `${data.seeds} seeds per row, primary metric per task.` : 'Loading…'}
            {' '}Compute sponsored by <b>Recrewty</b>.
          </p>
        </div>

        <div className="lb-controls">
          <div className="lb-ctl">
            <span className="lb-ctl-k">Language</span>
            <div className="lb-seg">
              {LEADERBOARDS.map((l) => (
                <button
                  key={`${l.benchmark}-${l.language}`}
                  type="button"
                  disabled={!l.available}
                  className={lang === l.language ? 'active' : ''}
                  title={l.available ? l.label : `${l.label} — coming in v0.2`}
                  onClick={() => l.available && setLang(l.language)}
                >
                  {l.language.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {data && (
            <div className="lb-ctl">
              <span className="lb-ctl-k">Rank by</span>
              <div className="lb-seg">
                <button
                  type="button"
                  className={rankBy === 'avg' ? 'active' : ''}
                  onClick={() => setRankBy('avg')}
                >
                  Avg
                </button>
                {data.ranked_tasks.map((t) => (
                  <button
                    key={t}
                    type="button"
                    className={rankBy === t ? 'active' : ''}
                    onClick={() => setRankBy(t)}
                  >
                    {TASK_LABELS[t] || t}
                  </button>
                ))}
              </div>
            </div>
          )}
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
                    <th
                      key={t}
                      className={`lb-num ${rankBy === t ? 'lb-col-active' : ''}`}
                      onClick={() => setRankBy(t)}
                    >
                      {TASK_LABELS[t] || t}
                      <span className="lb-metric">({data.task_primary_metrics[t]})</span>
                      {rankBy === t && <span className="lb-caret">▼</span>}
                    </th>
                  ))}
                  <th
                    className={`lb-num lb-avg ${rankBy === 'avg' ? 'lb-col-active' : ''}`}
                    onClick={() => setRankBy('avg')}
                  >
                    Avg
                    {rankBy === 'avg' && <span className="lb-caret">▼</span>}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row, i) => {
                  const rankByTaskValue = sortValue(row, rankBy)
                  const displayRank =
                    rankBy === 'avg'
                      ? row.rank ?? (row.partial_flag || '—')
                      : rankByTaskValue === null
                      ? '—'
                      : i + 1
                  return (
                    <tr key={row.model} className={!row.complete ? 'lb-partial' : ''}>
                      <td className="lb-rank">{displayRank}</td>
                      <td className="lb-model">
                        <div className="lb-model-name">{row.model}</div>
                        <div className="lb-model-id">{row.model_id}</div>
                      </td>
                      <td className="lb-params">{row.params_display}</td>
                      {data.ranked_tasks.map((t) => {
                        const { main, stdev } = formatCell(row.results[t])
                        return (
                          <td
                            key={t}
                            className={`lb-num ${rankBy === t ? 'lb-col-active' : ''}`}
                          >
                            <div className="lb-cell-main">{main}</div>
                            {stdev !== null && <div className="lb-cell-stdev">± {stdev}</div>}
                          </td>
                        )
                      })}
                      <td className={`lb-num lb-avg ${rankBy === 'avg' ? 'lb-col-active' : ''}`}>
                        <div className="lb-cell-main"><b>{row.avg.toFixed(2)}</b></div>
                        {!row.complete && <div className="lb-cell-stdev">{row.partial_flag}</div>}
                      </td>
                    </tr>
                  )
                })}
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
