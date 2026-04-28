import { useState } from 'react'
import Topbar from '../components/Topbar.jsx'
import Nav from '../components/Nav.jsx'
import Marquee from '../components/Marquee.jsx'
import Footer from '../components/Footer.jsx'

export default function Home() {
  const [email, setEmail] = useState('')
  const [note, setNote] = useState('~ occasional updates on new benchmarks, models, and leaderboard releases')
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
              <span className="chip">● LIVE NOW</span>
              <span>V1.0 · RELEASED 2026-04-28</span>
            </div>
            <h1>
              Every model,<br />
              <span className="stroke">measured</span><span className="slash">.</span>
            </h1>
            <p className="sub">
              An open, reproducible benchmark and leaderboard for language models across <b>Serbian, Montenegrin, Croatian</b> and <b>Bosnian</b>. Follow the release for new datasets, model evaluations, and benchmark expansions across the BCMS ecosystem.
            </p>
          </div>

          <div>
            <form className="signup" onSubmit={onSubmit}>
              <div className="signup-k">§ NOTIFY ME</div>
              <div className="signup-h">Get updates.</div>
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
              <span>BUILD / <b>v1.0</b></span>
              <span>RELEASED · PROD</span>
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
              <div className="progress-top"><span>LAUNCH READINESS</span><b>v1.0</b></div>
              <div className="progress-bar"></div>
            </div>

            <div className="term">
              <div><span className="g">$</span> <span className="w">balkanbench</span> <span className="y">--version</span></div>
              <div><span className="c">› v1.0 - 67,313 items across 3 released languages</span></div>
              <div><span className="c">› 9 models · 5 seeds · held-out test split</span></div>
              <div><span className="c">› results live at balkanbench.com/leaderboard</span></div>
              <div><span className="g">›</span> <span className="w">released</span> <span className="y">2026-04-28</span><span className="cursor"></span></div>
            </div>
          </div>
        </div>
      </section>

      <Marquee />
      <Footer />
    </>
  )
}
