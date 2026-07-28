import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react' // eslint-disable-line no-unused-vars
import { TASK_LABELS, TASK_DESCRIPTIONS } from '../lib/leaderboards.js'
import '../styles/controls.css'

const SPRING = { type: 'spring', bounce: 0, duration: 0.3 }

export default function RankMenu({ value, onChange, tasks, metrics }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    const onDown = (e) => { if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('keydown', onKey)
    document.addEventListener('pointerdown', onDown)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('pointerdown', onDown)
    }
  }, [open])

  const currentLabel = value === 'avg' ? 'Avg' : (TASK_LABELS[value] || value)
  const items = [
    { value: 'avg', label: 'Avg', metric: null, desc: 'Unweighted mean of the primary task scores' },
    ...tasks.map(t => ({ value: t, label: TASK_LABELS[t] || t, metric: metrics[t], desc: TASK_DESCRIPTIONS[t] || '' })),
  ]

  return (
    <div className="rankmenu" ref={rootRef}>
      <button
        type="button"
        className="rankmenu-pill"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(o => !o)}
      >
        Rank by: <b>{currentLabel}</b>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            role="menu"
            className="rankmenu-pop"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={SPRING}
          >
            {items.map(item => (
              <button
                key={item.value}
                type="button"
                role="menuitemradio"
                aria-checked={item.value === value}
                className={`rankmenu-item ${item.value === value ? 'active' : ''}`}
                onClick={() => { onChange(item.value); setOpen(false) }}
              >
                <span className="rankmenu-item-head">
                  {item.label}
                  {item.metric && <span className="rankmenu-metric num">{item.metric}</span>}
                </span>
                <span className="rankmenu-desc">{item.desc}</span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
