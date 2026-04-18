import { useEffect, useState } from 'react'
import './App.css'

const LAUNCH_DATE = new Date('2026-04-27T09:00:00Z')

function useCountdown(target) {
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const diff = Math.max(0, target.getTime() - now.getTime())
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  const hours = Math.floor((diff / (1000 * 60 * 60)) % 24)
  const minutes = Math.floor((diff / (1000 * 60)) % 60)
  const seconds = Math.floor((diff / 1000) % 60)
  return { days, hours, minutes, seconds }
}

const pad = (n) => String(n).padStart(2, '0')

function App() {
  const { days, hours, minutes, seconds } = useCountdown(LAUNCH_DATE)

  return (
    <div className="page">
      <div className="mesh mesh-1" />
      <div className="mesh mesh-2" />
      <div className="grain" />

      <main className="stage">
        <h1 className="headline">
          <span className="line line-1">Coming</span>
          <span className="line line-2">Soon.</span>
        </h1>

        <p className="lede">April 27, 2026</p>

        <div className="countdown" role="timer" aria-label="Launch countdown">
          {[
            { label: 'days', value: days },
            { label: 'hours', value: hours },
            { label: 'minutes', value: minutes },
            { label: 'seconds', value: seconds },
          ].map((b) => (
            <div className="cd-block" key={b.label}>
              <div className="cd-value">{pad(b.value)}</div>
              <div className="cd-label">{b.label}</div>
            </div>
          ))}
        </div>
      </main>

      <footer className="footer">
        <span className="powered-label">Powered by</span>
        <a
          href="https://recrewty.com"
          target="_blank"
          rel="noopener noreferrer"
          className="powered-mark"
        >
          <img src="/recrewty-logo-white.png" alt="Recrewty" />
        </a>
      </footer>
    </div>
  )
}

export default App
