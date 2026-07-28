import { useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'motion/react' // eslint-disable-line no-unused-vars
import { TASK_LABELS, TASK_DESCRIPTIONS, formatAvg } from '../lib/leaderboards.js'
import ScoreCell from './ScoreCell.jsx'
import '../styles/sheet.css'

const ENTRY = { type: 'spring', bounce: 0.2, duration: 0.4 }

export default function ModelSheet({ row, board, protocol, onClose }) {
  const panelRef = useRef(null)
  const restoreRef = useRef(null)

  useEffect(() => {
    if (!row) return
    restoreRef.current = document.activeElement
    panelRef.current?.focus()
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      restoreRef.current?.focus?.()
    }
  }, [row, onClose])

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
            aria-label={row.model}
            tabIndex={-1}
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '110%' }}
            transition={ENTRY}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0.2, bottom: 0.6 }}
            onDragEnd={(e, info) => {
              if (info.velocity.y > 300 || (info.velocity.y >= 0 && info.offset.y > 160)) onClose()
            }}
          >
            <div className="sheet-grab" aria-hidden="true" />
            <div className="sheet-head">
              <div>
                <h2 className="sheet-title">{row.model}</h2>
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
