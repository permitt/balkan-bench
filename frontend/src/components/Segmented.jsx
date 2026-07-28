import { useId } from 'react'
import { motion } from 'motion/react' // eslint-disable-line no-unused-vars
import '../styles/controls.css'

const SPRING = { type: 'spring', bounce: 0, duration: 0.3 }

// Finds the next non-disabled option in `dir` (1 or -1) from `fromIdx`,
// wrapping around the ends. Returns null if every option is disabled.
function nextEnabledOption(options, fromIdx, dir) {
  const n = options.length
  for (let step = 1; step <= n; step++) {
    const idx = ((fromIdx + dir * step) % n + n) % n
    if (!options[idx].disabled) return options[idx]
  }
  return null
}

export default function Segmented({ label, value, onChange, options }) {
  const groupId = useId()

  const handleKeyDown = (e) => {
    let dir = 0
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') dir = 1
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') dir = -1
    else return
    e.preventDefault()
    const currentIdx = options.findIndex(o => o.value === value)
    const next = nextEnabledOption(options, currentIdx, dir)
    if (next) onChange(next.value)
  }

  return (
    <div className="seg" role="radiogroup" aria-label={label} onKeyDown={handleKeyDown}>
      {options.map((opt) => {
        const active = opt.value === value
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={opt.disabled}
            title={opt.title}
            className={`seg-item ${active ? 'active' : ''}`}
            onClick={() => !opt.disabled && onChange(opt.value)}
          >
            {active && (
              <motion.span layoutId={`seg-ind-${groupId}`} className="seg-ind" transition={SPRING} />
            )}
            <span className="seg-label">{opt.label}</span>
            {opt.sublabel && <span className="seg-sub">{opt.sublabel}</span>}
            {opt.disabled && opt.badge && <span className="seg-badge">{opt.badge}</span>}
          </button>
        )
      })}
    </div>
  )
}
