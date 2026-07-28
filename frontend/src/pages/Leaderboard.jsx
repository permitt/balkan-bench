import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  BENCHMARKS, LANGUAGES, resolveBoard, boardEntriesFor,
} from '../lib/leaderboards.js'
import { useBoard } from '../lib/useBoard.js'
import Segmented from '../components/Segmented.jsx'
import RankMenu from '../components/RankMenu.jsx'
import LeaderboardTable from '../components/LeaderboardTable.jsx'
import ModelSheet from '../components/ModelSheet.jsx'
import Skeleton from '../components/Skeleton.jsx'
import '../styles/leaderboard.css'

export default function Leaderboard() {
  const [params, setParams] = useSearchParams()
  const bench = params.get('benchmark') || 'superglue'
  const lang = params.get('lang') || 'sr'
  const rankBy = params.get('task') || 'avg'
  const modelParam = params.get('model')

  const target = resolveBoard(bench, lang)
  const isSle = target.benchmark === 'sle'
  const { data, apiData, error, retry } = useBoard(target)

  const setParam = useCallback((key, value, defaultValue) => {
    const next = new URLSearchParams(params)
    if (value === defaultValue) next.delete(key)
    else next.set(key, value)
    setParams(next, { replace: true })
  }, [params, setParams])

  const closeSheet = useCallback(() => setParam('model', null, null), [setParam])
  const setBench = useCallback(v => setParam('benchmark', v, 'superglue'), [setParam])
  const setLang = useCallback(v => setParam('lang', v, null), [setParam])
  const setRankBy = useCallback(v => setParam('task', v, 'avg'), [setParam])
  const selectModel = useCallback(r => setParam('model', r.model, null), [setParam])

  const selectedRow = useMemo(() => {
    if (!modelParam) return null
    return (
      data?.rows.find(r => r.model === modelParam) ??
      apiData?.rows.find(r => r.model === modelParam) ??
      null
    )
  }, [modelParam, data, apiData])

  const selectedBoard = data?.rows.includes(selectedRow) ? data : apiData
  const selectedProtocol = !isSle || !selectedRow ? null
    : data?.rows.includes(selectedRow)
      ? 'Open weights - loglikelihood protocol'
      : 'Closed API - generative protocol'

  const benchOptions = Object.entries(BENCHMARKS).map(([key, meta]) => ({
    value: key,
    label: meta.label,
    sublabel: meta.tagline,
    disabled: !meta.available,
    badge: meta.availableIn,
    title: meta.description,
  }))

  const langOptions = boardEntriesFor(target.benchmark).map(entry => ({
    value: entry.language,
    label: `${LANGUAGES[entry.language].flag} ${entry.language.toUpperCase()}`,
    sublabel: LANGUAGES[entry.language].nativeName,
    disabled: !entry.available,
    badge: entry.availableIn,
  }))

  return (
    <section className="lb-page container">
      <header className="lb-head">
        <h1 className="display lb-title">Leaderboard</h1>
        <p className="lb-sub">
          {data?.seeds !== undefined
            ? `Mean ± stdev across ${data.seeds} seeds on the held-out test split.`
            : 'Evaluated on the held-out test split.'}
        </p>
      </header>

      <div className="lb-toolbar">
        <Segmented label="Benchmark" value={target.benchmark} onChange={setBench} options={benchOptions} />
        <Segmented label="Language" value={target.language} onChange={setLang} options={langOptions} />
        {data && (
          <RankMenu value={rankBy} onChange={setRankBy} tasks={data.ranked_tasks} metrics={data.task_primary_metrics} />
        )}
      </div>

      {error && (
        <div className="lb-error-card" role="alert">
          <p>Failed to load leaderboard: {error}</p>
          <button type="button" onClick={retry}>Retry</button>
        </div>
      )}

      {!data && !error && <Skeleton rows={8} cols={9} />}

      {data && isSle && (
        <>
          <p className="lb-note">The two boards use different scoring protocols and are not comparable to each other.</p>
          <h2 className="lb-board-heading">Open weights - loglikelihood protocol</h2>
          <LeaderboardTable data={data} rankBy={rankBy} onRankBy={setRankBy} onSelectModel={selectModel} />
          <h2 className="lb-board-heading">Closed API - generative protocol</h2>
          {apiData
            ? <LeaderboardTable data={apiData} rankBy={rankBy} onRankBy={setRankBy} onSelectModel={selectModel} />
            : <Skeleton rows={4} cols={9} />}
        </>
      )}

      {data && !isSle && (
        <LeaderboardTable data={data} rankBy={rankBy} onRankBy={setRankBy} onSelectModel={selectModel} />
      )}

      <ModelSheet
        row={selectedRow}
        board={selectedBoard ?? data}
        protocol={selectedProtocol}
        onClose={closeSheet}
      />
    </section>
  )
}
