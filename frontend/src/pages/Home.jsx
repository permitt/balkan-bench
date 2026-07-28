import { Link } from 'react-router-dom'
import { motion } from 'motion/react' // eslint-disable-line no-unused-vars
import { resolveBoard, sortRows, displayModelName, FACTS, BENCHMARKS } from '../lib/leaderboards.js'
import { useBoard } from '../lib/useBoard.js'
import ScoreCell from '../components/ScoreCell.jsx'
import '../styles/home.css'

const RISE = (i) => ({
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { type: 'spring', bounce: 0, duration: 0.4, delay: i * 0.05 },
})

function PreviewCard({ label, title, to, board }) {
  const top5 = board ? sortRows(board.rows, 'avg').slice(0, 5) : []
  return (
    <section className="home-preview" aria-label={label}>
      <div className="home-preview-card">
        <div className="home-preview-head">
          <span>{title}</span>
          <Link to={to} className="home-preview-more">Full leaderboard</Link>
        </div>
        {top5.map((row, i) => (
          <motion.div key={row.model} className="home-preview-row" {...RISE(i)}>
            <span className="num home-preview-rank">{i + 1}</span>
            <span className="home-preview-model">{displayModelName(row.model)}</span>
            <span className="num home-preview-params">{row.params_display}</span>
            <ScoreCell cell={{ mean: row.avg }} active />
          </motion.div>
        ))}
      </div>
    </section>
  )
}

export default function Home() {
  const { data: sleData } = useBoard(resolveBoard('sle', 'sr'))
  const { data: superglueData } = useBoard(resolveBoard('superglue', 'sr'))

  return (
    <div className="home container">
      <section className="home-hero">
        <h1 className="display home-title">Every model, measured.</h1>
        <p className="home-sub">
          The open benchmark and leaderboard for language models in Serbian,
          Croatian, Bosnian, and Montenegrin. Independent, reproducible, and
          free for anyone to verify - or beat.
        </p>
        <div className="home-ctas">
          <Link className="btn-primary" to="/leaderboard">View leaderboard</Link>
          <Link className="btn-quiet" to="/about">Read methodology</Link>
        </div>
      </section>

      <div className="home-previews">
        <PreviewCard
          label="Serbian-LLM-Eval top models"
          title="Serbian-LLM-Eval · open weights · top 5, live from the leaderboard"
          to="/leaderboard?benchmark=sle"
          board={sleData}
        />
        <PreviewCard
          label="SuperGLUE top models"
          title="SuperGLUE · Serbian · top 5, live from the leaderboard"
          to="/leaderboard"
          board={superglueData}
        />
      </div>

      <section className="home-facts" aria-label="Key figures">
        <div className="home-fact">
          <span className="num home-fact-n">{FACTS.items.toLocaleString('en-US')}</span>
          <span className="home-fact-k">evaluation items across two suites</span>
        </div>
        <div className="home-fact">
          <span className="num home-fact-n">{FACTS.modelCount}</span>
          <span className="home-fact-k">models evaluated</span>
        </div>
        <div className="home-fact">
          <span className="num home-fact-n">{FACTS.languageCount}</span>
          <span className="home-fact-k">languages live</span>
        </div>
      </section>

      <section className="home-vision">
        <h2>Why this exists</h2>
        <p>
          Nearly 20 million people speak the BCMS languages, yet multilingual
          model claims almost never get tested on them. Model cards cite
          English benchmarks; the region gets guesswork. BalkanBench replaces
          guesswork with evidence: one canonical score per model, task, and
          language, with every result traceable to the exact dataset, config,
          and seed that produced it.
        </p>
      </section>

      <section className="home-points" aria-label="Principles">
        <div className="home-point">
          <h3>Open by default</h3>
          <p>
            MIT-licensed code, public datasets on Hugging Face, and hidden
            test labels that keep the leaderboard honest.
          </p>
        </div>
        <div className="home-point">
          <h3>Reproducible to the seed</h3>
          <p>
            Held-out test splits, five seeds per encoder run, schema-validated
            configs. Rerun any number yourself.
          </p>
        </div>
        <div className="home-point">
          <h3>Built by the region</h3>
          <p>
            Annotators and linguists across the Balkans adapted every dataset
            by hand. <Link to="/submit">Add your model</Link> - or propose the
            next track.
          </p>
        </div>
      </section>

      <section className="home-roster" aria-label="Benchmark tracks">
        <div className="home-roster-head">
          <h2>The tracks</h2>
          <p>
            Encoder NLU and generative few-shot are live today. Embeddings and
            a human-judged arena are next.
          </p>
        </div>
        {Object.entries(BENCHMARKS).map(([key, meta]) => (
          <div key={key} className="home-track">
            <div>
              <div className="home-track-name">{meta.label}</div>
              <div className="home-track-tag">{meta.tagline}</div>
            </div>
            <span className={`home-track-status ${meta.available ? 'live' : 'planned'}`}>
              {meta.available ? 'Live' : `Planned - ${meta.availableIn}`}
            </span>
          </div>
        ))}
      </section>
    </div>
  )
}
