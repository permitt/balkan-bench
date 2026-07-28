import '../styles/content.css'
import '../styles/home.css'

export default function Submit() {
  return (
    <div className="content container">
      <header className="content-head">
        <h1 className="display">Add your model.</h1>
        <p className="content-lede">
          BalkanBench is open. Four kinds of contribution: a new benchmark (new dataset),
          a new task inside an existing benchmark, a new model, or a run (predictions)
          for an existing model / benchmark pair.
        </p>
      </header>

      <section>
        <h2>1. Open an issue</h2>
        <p>
          Pick the right template on the BalkanBench GitHub repo. Required fields
          include identity (public GitHub or Hugging Face handle), license, and
          contact. Anonymous submissions are not accepted for leaderboard rows.
        </p>
        <div className="content-ctarow">
          <a className="btn-primary" href="https://github.com/permitt/balkan-bench/issues/new/choose" target="_blank" rel="noopener noreferrer">
            Open an issue
          </a>
          <a className="btn-quiet" href="https://github.com/permitt/balkan-bench/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">
            Read CONTRIBUTING
          </a>
        </div>
      </section>

      <section>
        <h2>2. Open a PR</h2>
        <p>
          After issue triage a maintainer will invite you to open a PR with the
          required YAML configs (validated by JSON Schema in CI) or a scored result
          artifact. The <code>balkanbench validate-config</code> CLI runs the exact
          validation CI uses.
        </p>
      </section>

      <section>
        <h2>3. CI + review + merge</h2>
        <p>
          CI runs lint, type, tests, coverage, schema validation, and a
          reproducibility gate. Maintainers review identity and license. On merge,
          the contribution ships in the next minor release.
        </p>
      </section>

      <section>
        <h2>Submitting results for an existing model</h2>
        <pre className="content-code">
{`balkanbench predict \\
  --model <name> --benchmark <bench> --language <lang>

balkanbench submit results/local/ --out submission.json

# open a Submission issue with submission.json attached`}
        </pre>
      </section>

      <section>
        <h2>Sponsorship</h2>
        <p>
          Official compute for v1.0 is sponsored by <b>Recrewty</b>. Community
          submissions that use different compute may leave the sponsor field as-is or
          replace it with their own sponsor; the leaderboard renders per-row sponsor
          when it differs from the benchmark default.
        </p>
      </section>
    </div>
  )
}
