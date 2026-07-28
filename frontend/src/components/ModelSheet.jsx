import { useEffect, useRef } from 'react'
import { AnimatePresence, motion, useDragControls } from 'motion/react' // eslint-disable-line no-unused-vars
import { TASK_LABELS, TASK_DESCRIPTIONS, formatAvg, displayModelName } from '../lib/leaderboards.js'
import ScoreCell from './ScoreCell.jsx'
import '../styles/sheet.css'

const ENTRY = { type: 'spring', bounce: 0.2, duration: 0.4 }

const FOCUSABLE_SELECTOR = 'a[href], button, [tabindex]:not([tabindex="-1"])'

export default function ModelSheet({ row, board, protocol, onClose }) {
  const panelRef = useRef(null)
  const restoreRef = useRef(null)
  const dragControls = useDragControls()

  useEffect(() => {
    if (!row) return
    restoreRef.current = document.activeElement
    panelRef.current?.focus()
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)

    const behind = [
      document.querySelector('.shell-main'),
      document.querySelector('.shell-nav'),
      document.querySelector('.shell-footer'),
    ].filter(Boolean)
    behind.forEach(el => el.setAttribute('inert', ''))

    return () => {
      document.removeEventListener('keydown', onKey)
      behind.forEach(el => el.removeAttribute('inert'))
      restoreRef.current?.focus?.()
    }
  }, [row, onClose])

  const trapTab = (e) => {
    if (e.key !== 'Tab' || !panelRef.current) return
    const focusables = Array.from(panelRef.current.querySelectorAll(FOCUSABLE_SELECTOR))
    if (focusables.length === 0) return
    const first = focusables[0]
    const last = focusables[focusables.length - 1]
    const active = document.activeElement
    if (e.shiftKey) {
      if (active === first || !panelRef.current.contains(active)) {
        e.preventDefault()
        last.focus()
      }
    } else if (active === last) {
      e.preventDefault()
      first.focus()
    }
  }

  return (
    <AnimatePresence>
      {row && (
        <>
          <motion.div
            className="sheet-scrim"
            data-testid="sheet-scrim"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.div
            ref={panelRef}
            className="sheet-panel"
            role="dialog"
            aria-modal="true"
            aria-label={displayModelName(row.model)}
            tabIndex={-1}
            onKeyDown={trapTab}
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '110%' }}
            transition={ENTRY}
            drag="y"
            dragListener={false}
            dragControls={dragControls}
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0.2, bottom: 0.6 }}
            onDragEnd={(e, info) => {
              if (info.velocity.y > 300 || (info.velocity.y >= 0 && info.offset.y > 160)) onClose()
            }}
          >
            <div
              className="sheet-grab"
              aria-hidden="true"
              onPointerDown={(e) => dragControls.start(e)}
            />
            <div className="sheet-head" onPointerDown={(e) => dragControls.start(e)}>
              <div>
                <h2 className="sheet-title">{displayModelName(row.model)}</h2>
                {row.model_id && row.model_id.includes('/') ? (
                  <a
                    className="sheet-hf"
                    href={`https://huggingface.co/${row.model_id}`}
                    target="_blank" rel="noopener noreferrer"
                  >
                    {row.model_id}
                  </a>
                ) : <span className="sheet-hf">{row.model_id}</span>}
              </div>
              <button type="button" className="sheet-close" aria-label="Close" onClick={onClose}>×</button>
            </div>
            <div className="sheet-facts">
              <span className="num">{row.params_display}</span>
              <span className="num">avg {formatAvg(row)}</span>
              {protocol && <span>{protocol}</span>}
            </div>
            <ul className="sheet-tasks">
              {board.ranked_tasks.map(t => (
                <li key={t} className="sheet-task">
                  <div className="sheet-task-info">
                    <span className="sheet-task-name">{TASK_LABELS[t] || t}</span>
                    <span className="sheet-task-metric num">{board.task_primary_metrics[t]}</span>
                    <span className="sheet-task-desc">{TASK_DESCRIPTIONS[t] || ''}</span>
                  </div>
                  <ScoreCell cell={row.results[t] ?? null} active />
                </li>
              ))}
            </ul>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
