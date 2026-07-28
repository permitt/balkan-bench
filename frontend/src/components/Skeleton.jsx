export default function Skeleton({ rows = 6, cols = 8 }) {
  return (
    <div className="skeleton" data-testid="skeleton" aria-hidden="true">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="skeleton-row">
          {Array.from({ length: cols }, (_, c) => (
            <span key={c} className="skeleton-cell" />
          ))}
        </div>
      ))}
    </div>
  )
}
