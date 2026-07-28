import { useId } from 'react'
import { motion } from 'motion/react' // eslint-disable-line no-unused-vars
import '../styles/controls.css'

const SPRING = { type: 'spring', bounce: 0, duration: 0.3 }

export default function Segmented({ label, value, onChange, options }) {
  const groupId = useId()
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
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
