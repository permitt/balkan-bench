import { formatCell } from '../lib/leaderboards.js'

export default function ScoreCell({ cell, active }) {
  const { main, stdev } = formatCell(cell)
  const hasBar = cell !== null && cell !== undefined
  return (
    <div className={`score ${active ? 'active' : ''}`}>
      <div className="score-main num">{main}</div>
      {stdev !== null && <div className="score-stdev num">± {stdev}</div>}
      {hasBar && (
        <div className="score-bar">
          <div
            className="score-bar-fill"
            data-testid="score-bar-fill"
            style={{ width: `${parseFloat((Number(cell.mean) * 100).toFixed(2))}%` }}
          />
        </div>
      )}
    </div>
  )
}
