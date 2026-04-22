import { useEffect, useState } from 'react'
import Topbar from '../components/Topbar.jsx'
import Nav from '../components/Nav.jsx'
import Marquee from '../components/Marquee.jsx'
import Footer from '../components/Footer.jsx'

const LAUNCH_DATE = new Date('2026-04-27T09:00:00Z')

function useCountdown(target) {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  const diff = Math.max(0, target.getTime() - now.getTime())
  const s = Math.floor(diff / 1000)
  return {
    days: Math.floor(s / 86400),
    hours: Math.floor((s % 86400) / 3600),
    minutes: Math.floor((s % 3600) / 60),
    seconds: s % 60,
  }
}

const pad = (n) => String(n).padStart(2, '0')

export default function Home() {
  const { days, hours, minutes, seconds } = useCountdown(LAUNCH_DATE)
  const [email, setEmail] = useState('')
  const [note, setNote] = useState('~ 420 researchers on the list')
  const [sent, setSent] = useState(false)

  const onSubmit = (e) => {
    e.preventDefault()
    if (!email.includes('@')) return
    setNote(`✓ ADDED · WE'LL EMAIL ${email.toUpperCase()} AT LAUNCH`)
    setSent(true)
    setEmail('')
  }

  return (
    <>
      <Topbar />
      <Nav />

      <section className="wrap">
        <div className="left">
          <div>
            <div className="eyebrow">
              <span className="chip">● COMING SOON</span>
              <span>V1.2 · ESTIMATED LAUNCH Q2 2026</span>
            </div>
            <h1>
              Every model,<br />
              <span className="stroke">measured</span><span className="slash">.</span><br />
              <span>Soon<span className="dot-acc">.</span></span>
            </h1>
            <p className="sub">
              An open, reproducible evaluation suite for LLMs across <b>Serbian, Montenegrin, Croatian</b> and <b>Bosnian</b>. We're finalizing datasets, pipelines and the public leaderboard - drop your email and we'll ping you the day it ships.
            </p>
          </div>

          <div>
            <div className="countdown">
              <div className="cd-cell"><div className="cd-k">Days</div><div className="cd-v">{pad(days)}</div></div>
              <div className="cd-cell"><div className="cd-k">Hours</div><div className="cd-v">{pad(hours)}</div></div>
              <div className="cd-cell"><div className="cd-k">Minutes</div><div className="cd-v">{pad(minutes)}</div></div>
              <div className="cd-cell"><div className="cd-k">Seconds</div><div className="cd-v">{pad(seconds)}</div></div>
            </div>

            <form className="signup" onSubmit={onSubmit}>
              <div className="signup-k">§ NOTIFY ME</div>
              <div className="signup-h">Get launch access.</div>
              <div className="signup-p">Early access to the leaderboard, eval harness, and submission pipeline. No spam, one email at launch.</div>
              <div className="signup-row">
                <input
                  type="email"
                  required
                  placeholder="your@email.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
                <button type="submit">
                  Notify me
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <path d="M5 12h14M13 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
              <div className="signup-note" style={sent ? { color: 'var(--signal)' } : undefined}>{note}</div>
            </form>
          </div>
        </div>

        <div className="right">
          <div className="status-card">
            <div className="status-head">
              <span>BUILD / <b>v1.2</b></span>
              <span>BUILDING · PROD</span>
            </div>

            <div className="status-body">
              <div className="sb-row">
                <div className="sb-n">01</div>
                <div className="sb-label">SuperGLUE · BCMS<span className="sub">Encoder NLU · 6 tasks</span></div>
                <div className="sb-status live">READY</div>
              </div>
              <div className="sb-row">
                <div className="sb-n">02</div>
                <div className="sb-label">Serbian-LLM-Eval<span className="sub">Generative · 7 tasks · OZ-Eval</span></div>
                <div className="sb-status prog">BETA</div>
              </div>
              <div className="sb-row">
                <div className="sb-n">03</div>
                <div className="sb-label">MTEB-BCMS<span className="sub">Embeddings · 4 tasks</span></div>
                <div className="sb-status planned">PLANNED</div>
              </div>
              <div className="sb-row">
                <div className="sb-n">04</div>
                <div className="sb-label">LLM Arena<span className="sub">Human-judged Elo</span></div>
                <div className="sb-status planned">PLANNED</div>
              </div>
            </div>

            <div className="progress">
              <div className="progress-top"><span>LAUNCH READINESS</span><b>62%</b></div>
              <div className="progress-bar"></div>
            </div>

            <div className="term">
              <div><span className="g">$</span> <span className="w">balkanbench</span> <span className="y">--pre-release</span></div>
              <div><span className="c">› loading 28,104 items across 4 languages</span></div>
              <div><span className="c">› validating 42 model baselines</span></div>
              <div><span className="c">› building leaderboard ui</span></div>
              <div><span className="g">›</span> <span className="w">eta</span> <span className="y">Q2 2026</span><span className="cursor"></span></div>
            </div>
          </div>
        </div>
      </section>

      <Marquee />
      <Footer />
    </>
  )
}
