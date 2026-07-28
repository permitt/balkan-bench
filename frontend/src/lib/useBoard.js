import { useCallback, useEffect, useState } from 'react'
import { boardUrl } from './leaderboards.js'

async function fetchJson(url) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export function useBoard(target) {
  const [data, setData] = useState(null)
  const [apiData, setApiData] = useState(null)
  const [error, setError] = useState(null)
  const [attempt, setAttempt] = useState(0)

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    let cancelled = false
    setData(null)
    setApiData(null)
    setError(null)

    fetchJson(boardUrl(target.path))
      .then((d) => { if (!cancelled) setData(d) })
      .catch((e) => { if (!cancelled) setError(e.message) })

    if (target.benchmark === 'sle') {
      fetchJson(boardUrl(`${target.path}-api`))
        .then((d) => { if (!cancelled) setApiData(d) })
        .catch(() => { if (!cancelled) setApiData({ rows: [] }) })
    }

    return () => { cancelled = true }
  }, [target.path, target.benchmark, attempt])
  /* eslint-enable react-hooks/set-state-in-effect */

  const retry = useCallback(() => setAttempt(a => a + 1), [])

  return { data, apiData, error, retry }
}
