import Topbar from '../components/Topbar.jsx'
import Nav from '../components/Nav.jsx'
import Footer from '../components/Footer.jsx'
import '../styles/content.css'

export default function Submit() {
  return (
    <>
      <Topbar />
      <Nav />
      <section className="content-wrap">
        <div className="eyebrow">
          <span className="chip">SUBMIT</span>
          <span>MODEL · BENCHMARK · RESULTS</span>
        </div>
        <h1 className="content-title">
          Add your <span className="stroke">model</span><span className="slash">.</span>
        </h1>
        <p className="content-lede">
          BalkanBench is open. Four kinds of contribution: a new benchmark (new dataset),
          a new task inside an existing benchmark, a new model, or a run (predictions)
          for an existing model / benchmark pair.
        </p>

        <h2>1. Open an issue</h2>
        <p>
          Pick the right template on the BalkanBench GitHub repo. Required fields
          include identity (public GitHub or Hugging Face handle), license, and
          contact. Anonymous submissions are not accepted for leaderboard rows.
        </p>
        <div className="content-ctarow">
          <a className="btn" href="https://github.com/permitt/balkan-bench/issues/new/choose" target="_blank" rel="noopener noreferrer">
            Open an issue
          </a>
          <a className="btn btn-ghost" href="https://github.com/permitt/balkan-bench/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">
            Read CONTRIBUTING
          </a>
        </div>

        <h2>2. Open a PR</h2>
        <p>
          After issue triage a maintainer will invite you to open a PR with the
          required YAML configs (validated by JSON Schema in CI) or a scored result
          artifact. The <code>balkanbench validate-config</code> CLI runs the exact
          validation CI uses.
        </p>

        <h2>3. CI + review + merge</h2>
        <p>
          CI runs lint, type, tests, coverage, schema validation, and a
          reproducibility gate. Maintainers review identity and license. On merge,
          the contribution ships in the next minor release.
        </p>

        <h2>Submitting results for an existing model</h2>
        <pre className="code">
{`balkanbench predict \\
  --model <name> --benchmark <bench> --language <lang>

balkanbench submit results/local/ --out submission.json

# open a Submission issue with submission.json attached`}
        </pre>

        <h2>Sponsorship</h2>
        <p>
          Official compute for v0.1 is sponsored by <b>Recrewty</b>. Community
          submissions that use different compute may leave the sponsor field as-is or
          replace it with their own sponsor; the leaderboard renders per-row sponsor
          when it differs from the benchmark default.
        </p>
      </section>
      <Footer />
    </>
  )
}
