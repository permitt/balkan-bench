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
  arc_challenge: 'ARC-C',
  arc_easy: 'ARC-E',
  hellaswag: 'HellaSwag',
  nq_open: 'NQ',
  openbookqa: 'OBQA',
  piqa: 'PIQA',
  triviaqa: 'TriviaQA',
  winogrande: 'WinoG',
}

const TASK_DESCRIPTIONS = {
  boolq: 'Boolean questions over a short passage',
  cb: 'Three-way textual entailment (entailment / contradiction / neutral)',
  copa: 'Choice of plausible alternatives (cause / effect)',
  rte: 'Binary textual entailment',
  multirc: 'Multi-sentence reading comprehension (grouped F1 + exact match)',
  wsc: 'Winograd Schema coreference reformulated as binary classification',
  arc_challenge: 'Grade-school science questions, challenge split, Serbian adaptation',
  arc_easy: 'Grade-school science questions, easy split, Serbian adaptation',
  hellaswag: 'Commonsense sentence completion, Serbian adaptation',
  nq_open: 'Open-domain question answering, Serbian adaptation',
  openbookqa: 'Open-book science QA over elementary-level facts, Serbian adaptation',
  piqa: 'Physical commonsense reasoning between two candidate solutions, Serbian adaptation',
  triviaqa: 'Trivia question answering, Serbian adaptation',
  winogrande: 'Winograd-style pronoun resolution, Serbian adaptation',
}

const LANGUAGES = {
  sr:  { flag: '🇷🇸', name: 'Serbian',     nativeName: 'Srpski' },
  hr:  { flag: '🇭🇷', name: 'Croatian',    nativeName: 'Hrvatski' },
  mne: { flag: '🇲🇪', name: 'Montenegrin', nativeName: 'Crnogorski' },
  bs:  { flag: '🇧🇦', name: 'Bosnian',     nativeName: 'Bosanski' },
}

const BENCHMARKS = {
  superglue: {
    label: 'SuperGLUE',
    tagline: 'Encoder NLU · 6 ranked tasks',
    description: 'Encoder-fine-tune NLU, 6 ranked tasks + 2 diagnostics.',
    available: true,
    availableIn: null,
  },
  sle: {
    label: 'Serbian-LLM-Eval',
    tagline: 'Generative few-shot',
    description: 'Generative few-shot eval (Aleksa Gordić) - ARC, HellaSwag, PIQA, BoolQ, Winogrande, etc.',
    available: true,
    availableIn: null,
  },
  mteb_bcms: {
    label: 'MTEB-BCMS',
    tagline: 'Embeddings · 4 tasks',
    description: 'Massive Text Embedding Benchmark, BCMS adaptation.',
    available: false,
    availableIn: 'v1.2',
  },
  llm_arena: {
    label: 'LLM Arena',
    tagline: 'Human-judged Elo',
    description: 'Head-to-head human preference ratings across BCMS LLMs.',
    available: false,
    availableIn: 'v1.2',
  },
}

// Discoverable leaderboards. When a new (benchmark, language) pair publishes
// its benchmark_results.json, flip `available: true`; no other code changes.
const LEADERBOARDS = [
  { benchmark: 'superglue', language: 'sr',  path: 'superglue-sr',  available: true,  availableIn: null   },
  { benchmark: 'superglue', language: 'hr',  path: 'superglue-hr',  available: true,  availableIn: null   },
  { benchmark: 'superglue', language: 'mne', path: 'superglue-mne', available: true,  availableIn: null   },
  { benchmark: 'superglue', language: 'bs',  path: 'superglue-bs',  available: false, availableIn: 'v1.1' },
  { benchmark: 'sle',       language: 'sr',  path: 'sle-sr',        available: true,  availableIn: null   },
]

function chipLabel(entry) {
  const lang = LANGUAGES[entry.language]
  const bench = BENCHMARKS[entry.benchmark]
  const tag = `${bench.label} · ${lang.name}`
  return entry.available ? tag : `${tag} - coming in ${entry.availableIn}`
}

function formatCell(cell) {
  if (cell === null || cell === undefined) return { main: '-', stdev: null }
  // Artifacts store sklearn-native 0-1 metric values; we render as 0-100
  // for human readability. Doing the rescale at display time keeps the
  // on-disk artifacts consistent with sklearn so that anyone reading
  // result.json directly gets the canonical metric value.
  const mean = (Number(cell.mean) * 100).toFixed(2)
  const stdev = cell.stdev === undefined ? null : (Number(cell.stdev) * 100).toFixed(2)
  return { main: mean, stdev }
}

function sortValue(row, rankBy) {
  if (rankBy === 'avg') return row.avg ?? null
  const cell = row.results[rankBy]
  return cell ? cell.mean : null
}

function sortRows(rows, rankBy) {
  const sorted = [...rows]
  sorted.sort((a, b) => {
    const av = sortValue(a, rankBy)
    const bv = sortValue(b, rankBy)
    if (av === null && bv === null) return 0
    if (av === null) return 1
    if (bv === null) return -1
    return bv - av
  })
  return sorted
}

// Renders one leaderboard table (+ its meta footer) from a benchmark_results.json
// payload. Shared by every (benchmark, language) pair, including the two
// access-split boards of the sle track.
function LeaderboardTable({ data, rankBy, setRankBy }) {
  const sortedRows = useMemo(() => sortRows(data.rows, rankBy), [data, rankBy])

  if (data.rows.length === 0) {
    return <div className="lb-empty">No results published yet.</div>
  }

  return (
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
                ? row.rank ?? (row.partial_flag || '-')
                : rankByTaskValue === null
                ? '-'
                : i + 1
            return (
              <tr key={row.model} className={!row.complete ? 'lb-partial' : ''}>
                <td className="lb-rank">{displayRank}</td>
                <td className="lb-model">
                  <div className="lb-model-name">{row.model}</div>
                  <div className="lb-model-id">
                    {row.model_id && row.model_id.includes('/') ? (
                      <a
                        href={`https://huggingface.co/${row.model_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {row.model_id}
                      </a>
                    ) : (
                      row.model_id
                    )}
                  </div>
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
                  <div className="lb-cell-main"><b>{(row.avg * 100).toFixed(2)}</b></div>
                  {!row.complete && <div className="lb-cell-stdev">{row.partial_flag}</div>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="lb-meta">
        <span>Benchmark version <b>{data.benchmark_version}</b></span>
        {data.seeds !== undefined && (
          <>
            <span className="sep">/</span>
            <span>{data.seeds} seeds</span>
          </>
        )}
        <span className="sep">/</span>
        <span>Generated {new Date(data.generated_at).toISOString().slice(0, 10)}</span>
        <span className="sep">/</span>
        <span>Compute sponsored by <b>{data.sponsor}</b></span>
      </div>
    </div>
  )
}

export default function Leaderboard() {
  const [params, setParams] = useSearchParams()
  const bench = params.get('benchmark') || 'superglue'
  const lang = params.get('lang') || 'sr'
  const rankBy = params.get('task') || 'avg'

  const target =
    LEADERBOARDS.find((l) => l.benchmark === bench && l.language === lang && l.available) ??
    LEADERBOARDS.find((l) => l.available) ??
    LEADERBOARDS[0]

  const langEntries = LEADERBOARDS.filter((l) => l.benchmark === target.benchmark)
  const isSle = target.benchmark === 'sle'

  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  // Only used for the sle track's second (closed-API) board. A missing/404
  // export is treated as "no results yet", not as a page-breaking error.
  const [apiData, setApiData] = useState(null)

  useEffect(() => {
    setData(null)
    setError(null)
    setApiData(null)
    fetch(`/leaderboards/${target.path}/benchmark_results.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setData)
      .catch((e) => setError(e.message))

    if (target.benchmark === 'sle') {
      fetch(`/leaderboards/${target.path}-api/benchmark_results.json`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`)
          return r.json()
        })
        .then(setApiData)
        .catch(() => setApiData({ rows: [] }))
    }
  }, [target.path, target.benchmark])

  const setLang = (nextLang) => {
    const next = new URLSearchParams(params)
    next.set('lang', nextLang)
    setParams(next, { replace: true })
  }

  const setBench = (nextBench) => {
    const next = new URLSearchParams(params)
    if (nextBench === 'superglue') next.delete('benchmark')
    else next.set('benchmark', nextBench)
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
      <section className="lb-page">
        <aside className="lb-sidebar">
          <div className="lb-side-k">Benchmarks</div>
          <nav className="lb-side-nav">
            {Object.entries(BENCHMARKS).map(([key, meta]) => {
              const isActive = target.benchmark === key
              const title = meta.available
                ? `${meta.label} - ${meta.description}`
                : `${meta.label} - coming in ${meta.availableIn}. ${meta.description}`
              return (
                <button
                  key={key}
                  type="button"
                  className={`lb-side-item ${isActive ? 'active' : ''} ${!meta.available ? 'disabled' : ''}`}
                  title={title}
                  aria-label={title}
                  disabled={!meta.available}
                  onClick={() => meta.available && setBench(key)}
                >
                  <div className="lb-side-label">
                    {meta.label}
                    {!meta.available && <span className="lb-side-soon">{meta.availableIn}</span>}
                  </div>
                  <div className="lb-side-tag">{meta.tagline}</div>
                </button>
              )
            })}
          </nav>
          <div className="lb-side-footnote">
            New benchmark?{' '}
            <a href="https://github.com/permitt/balkan-bench/issues/new/choose" target="_blank" rel="noopener noreferrer">
              Propose one
            </a>
            . See{' '}
            <a href="https://github.com/permitt/balkan-bench/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">
              CONTRIBUTING
            </a>.
          </div>
        </aside>

        <div className="lb-main">
        <div className="lb-head">
          <div className="lb-eyebrow">
            <span className="chip">LEADERBOARD</span>
            <span>
              {BENCHMARKS[target.benchmark].label.toUpperCase()} ·{' '}
              {LANGUAGES[target.language].flag} {LANGUAGES[target.language].name.toUpperCase()} · V1.0
            </span>
            {data && data.seeds !== undefined && (
              <span className="chip chip-test" title={`Numbers reported are mean ± stdev across ${data.seeds} seeds, evaluated on the held-out test split`}>
                TEST · {data.seeds} SEEDS
              </span>
            )}
          </div>
          <h1 className="lb-title">
            Every model, <span className="stroke">measured</span><span className="slash">.</span>
          </h1>
          <p className="lb-sub">
            {data
              ? data.seeds !== undefined
                ? `${data.seeds} seeds per row, evaluated on the held-out test split, mean ± stdev shown.`
                : 'Single run per row, evaluated on the held-out test split.'
              : 'Loading…'}
            {' '}Compute sponsored by <b>Recrewty</b>.
          </p>
        </div>

        <div className="lb-controls">
          <div className="lb-ctl">
            <span className="lb-ctl-k">Language</span>
            <div className="lb-seg">
              {langEntries.map((l) => {
                const langMeta = LANGUAGES[l.language]
                return (
                  <button
                    key={`${l.benchmark}-${l.language}`}
                    type="button"
                    disabled={!l.available}
                    className={lang === l.language ? 'active' : ''}
                    title={chipLabel(l)}
                    aria-label={chipLabel(l)}
                    onClick={() => l.available && setLang(l.language)}
                  >
                    <span className="lb-flag" aria-hidden="true">{langMeta.flag}</span>
                    <span className="lb-lang-code">{l.language.toUpperCase()}</span>
                    <span className="lb-lang-name">{langMeta.nativeName}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {data && (
            <div className="lb-ctl">
              <span className="lb-ctl-k">Rank by</span>
              <div className="lb-seg">
                <button
                  type="button"
                  className={rankBy === 'avg' ? 'active' : ''}
                  title={`Main score - unweighted mean of the ${data.ranked_tasks.length} primary task scores`}
                  onClick={() => setRankBy('avg')}
                >
                  Avg
                </button>
                {data.ranked_tasks.map((t) => (
                  <button
                    key={t}
                    type="button"
                    className={rankBy === t ? 'active' : ''}
                    title={`${TASK_LABELS[t] || t} (${data.task_primary_metrics[t]}) - ${TASK_DESCRIPTIONS[t] || ''}`}
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

        {isSle ? (
          <>
            <p className="lb-note">
              The two boards use different scoring protocols and are not comparable to each other.
            </p>
            <h2 className="lb-board-heading">Open weights - loglikelihood protocol</h2>
            {data && <LeaderboardTable data={data} rankBy={rankBy} setRankBy={setRankBy} />}
            <h2 className="lb-board-heading">Closed API - generative protocol</h2>
            {apiData ? (
              <LeaderboardTable data={apiData} rankBy={rankBy} setRankBy={setRankBy} />
            ) : (
              <div className="lb-loading">Loading leaderboard…</div>
            )}
          </>
        ) : (
          data && <LeaderboardTable data={data} rankBy={rankBy} setRankBy={setRankBy} />
        )}

        {!data && !error && <div className="lb-loading">Loading leaderboard…</div>}
        </div>
      </section>
      <Footer />
    </>
  )
}
