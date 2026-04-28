import Topbar from '../components/Topbar.jsx'
import Nav from '../components/Nav.jsx'
import Footer from '../components/Footer.jsx'
import '../styles/content.css'

export default function About() {
  return (
    <>
      <Topbar />
      <Nav />
      <section className="content-wrap">
        <div className="eyebrow">
          <span className="chip">ABOUT</span>
          <span>METHODOLOGY · SCOPE · SPONSOR</span>
        </div>
        <h1 className="content-title">
          Why <span className="stroke">BalkanBench</span>?
        </h1>
        <p className="content-lede">
          An open, reproducible, auditable benchmark for language models evaluated on
          Serbian, Croatian, Montenegrin, and Bosnian. The goal is one canonical number
          per (model, task, language) with every result traceable back to the exact
          dataset, config, and seed that produced it.
        </p>

        <h2>What ships in v1.0</h2>
        <ul>
          <li>
            <b>Serbian SuperGLUE</b> (official frozen track): 6 ranked tasks
            (BoolQ, CB, COPA, RTE, MultiRC, WSC) and 2 diagnostics (AX-b, AX-g),
            totalling <b>67,313 items</b> across train, validation, and held-out
            test splits.
          </li>
          <li>
            <b>Croatian + Montenegrin SuperGLUE</b> as released previews: 5 ranked
            tasks each (no WSC adaptation yet); HR/MNE rows publish on the same
            leaderboard, scored on the same private test labels.
          </li>
          <li>9 baseline encoder models evaluated with 5 fixed seeds each on the held-out test split.</li>
          <li>Public Hugging Face datasets with hidden test labels held in gated sibling repos.</li>
          <li>Open-source framework for local + GCP (Vertex AI) evaluation.</li>
        </ul>

        <h2>What's next</h2>
        <ul>
          <li>Bosnian SuperGLUE adaptation (sibling HF repo).</li>
          <li>Serbian-LLM-Eval (Aleksa Gordić) as a second benchmark suite.</li>
          <li>MTEB-BCMS embeddings, LLM Arena, and community-submitted tracks (sentiment, NER, domain-specific).</li>
        </ul>

        <h2>Hidden test labels</h2>
        <p>
          Public users run <code>balkanbench predict</code> locally to generate
          predictions on the unlabeled test split. Official scoring happens in a
          trusted environment via <code>balkanbench score</code> against a private
          labels HF repo. This preserves leaderboard integrity.
        </p>

        <h2>Sponsor</h2>
        <p>
          Compute for the official v1.0 evaluation is sponsored by{' '}
          <a href="https://recrewty.com" target="_blank" rel="noopener noreferrer">
            <b>Recrewty</b>
          </a>. Every result artifact and the leaderboard export carry this acknowledgement.
        </p>

        <h2>Further reading</h2>
        <ul>
          <li><a href="https://medium.com/@permitt/release-of-balkanbench-vision-behind-it-fd1ba73be411" target="_blank" rel="noopener noreferrer">Release of BalkanBench: the vision behind it (Medium)</a></li>
          <li><a href="https://github.com/permitt/balkan-bench" target="_blank" rel="noopener noreferrer">Source code on GitHub</a></li>
          <li><a href="https://github.com/permitt/balkan-bench/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">How to contribute</a></li>
          <li><a href="https://huggingface.co/datasets/permitt/superglue-sr" target="_blank" rel="noopener noreferrer">Serbian SuperGLUE dataset</a></li>
        </ul>
      </section>
      <Footer />
    </>
  )
}
