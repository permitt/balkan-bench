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

        <h2>What ships in v0.1</h2>
        <ul>
          <li>SuperGLUE adapted for Serbian, 6 ranked tasks, AXb + AXg diagnostics.</li>
          <li>9 baseline models evaluated with 5 fixed seeds each.</li>
          <li>Public Hugging Face dataset with hidden test labels held privately.</li>
          <li>Open-source framework for local + GCP evaluation.</li>
        </ul>

        <h2>What's next</h2>
        <ul>
          <li>Croatian, Montenegrin, Bosnian datasets (sibling HF repos).</li>
          <li>Serbian-LLM-Eval (Aleksa Gordić) as a second benchmark suite.</li>
          <li>MTEB-BCMS embeddings and LLM Arena.</li>
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
          Compute for the official v0.1 evaluation is sponsored by{' '}
          <a href="https://recrewty.com" target="_blank" rel="noopener noreferrer">
            <b>Recrewty</b>
          </a>. Every result artifact and the leaderboard export carry this acknowledgement.
        </p>

        <h2>Further reading</h2>
        <ul>
          <li><a href="https://github.com/permitt/balkan-bench/blob/main/docs/superpowers/specs/2026-04-22-balkanbench-v0.1-design.md">v0.1 Design Spec</a></li>
          <li><a href="https://github.com/permitt/balkan-bench/blob/main/docs/methodology/data_provenance.md">Data Provenance</a></li>
          <li><a href="https://github.com/permitt/balkan-bench/blob/main/docs/methodology/throughput.md">Throughput Methodology</a></li>
          <li><a href="https://github.com/permitt/balkan-bench/blob/main/CONTRIBUTING.md">How to Contribute</a></li>
        </ul>
      </section>
      <Footer />
    </>
  )
}
