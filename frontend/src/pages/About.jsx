import { Link } from 'react-router-dom'
import '../styles/content.css'

export default function About() {
  return (
    <div className="content container">
      <header className="content-head">
        <h1 className="display">Why BalkanBench?</h1>
        <p className="content-lede">
          An open, reproducible, auditable benchmark for language models evaluated on
          Serbian, Croatian, Montenegrin, and Bosnian. The goal is one canonical number
          per (model, task, language) with every result traceable back to the exact
          dataset, config, and seed that produced it.
        </p>
        <nav className="content-nav" aria-label="Sections">
          <Link to="/about#benchmarks">Benchmarks</Link>
          <Link to="/about#methodology">Methodology</Link>
          <Link to="/about#acknowledgements">Acknowledgements</Link>
        </nav>
      </header>

      <section id="benchmarks">
        <h2>Benchmarks</h2>
        <h3>What ships in v1.0</h3>
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

        <h3>What's next</h3>
        <ul>
          <li>Bosnian SuperGLUE adaptation (sibling HF repo).</li>
          <li>Serbian-LLM-Eval (Aleksa Gordić) as a second benchmark suite.</li>
          <li>MTEB-BCMS embeddings, LLM Arena, and community-submitted tracks (sentiment, NER, domain-specific).</li>
        </ul>
      </section>

      <section id="methodology">
        <h2>Methodology</h2>
        <h3>Test evaluation protocol</h3>
        <p>
          Public users run <code>balkanbench predict</code> locally to generate
          predictions on the unlabeled test split. Official scoring happens in a
          trusted environment via <code>balkanbench score</code> against a private
          labels HF repo. This preserves leaderboard integrity.
        </p>

        <h3>Sponsor</h3>
        <p>
          Compute for the official v1.0 evaluation is sponsored by{' '}
          <a href="https://recrewty.com" target="_blank" rel="noopener noreferrer">
            <b>Recrewty</b>
          </a>. Every result artifact and the leaderboard export carry this acknowledgement.
        </p>

        <h3>Further reading</h3>
        <ul>
          <li><a href="https://medium.com/@permitt/release-of-balkanbench-vision-behind-it-fd1ba73be411" target="_blank" rel="noopener noreferrer">Release of BalkanBench: the vision behind it (Medium)</a></li>
          <li><a href="https://github.com/permitt/balkan-bench" target="_blank" rel="noopener noreferrer">Source code on GitHub</a></li>
          <li><a href="https://github.com/permitt/balkan-bench/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">How to contribute</a></li>
          <li><a href="https://huggingface.co/datasets/permitt/superglue-sr" target="_blank" rel="noopener noreferrer">Serbian SuperGLUE dataset</a></li>
        </ul>
      </section>

      <section id="acknowledgements">
        <h2>Acknowledgements</h2>
        <p>
          Contributors can propose an entirely new benchmark, extend an
          existing one with additional tasks, add a model configuration for
          leaderboard evaluation, or submit results for an already supported
          setup. The repository includes clear documentation and a
          schema-validated PR workflow so that new contributions can be
          reviewed transparently and integrated in a standardized way. If you
          are working on BCMS language evaluation resources, sentiment or NER
          datasets, retrieval benchmarks, or new language localizations, this
          project is meant to be a place where that work can live publicly
          and be compared reproducibly. Please note also that the eval has
          been done in combination with hyperparameter search in Optuna for
          every model, but if you want a specific config, propose it so we
          rerun your model on the test eval.
        </p>

        <p>This work wouldn't have been possible without the people who made it real.</p>
        <ul>
          <li>
            <b>Daria Milošević</b> and <b>Nina Škoro</b> annotated the Serbian
            datasets.
          </li>
          <li>
            <b>Nikola Ljubešić</b>, <b>Mirna Potočnjak</b>, and{' '}
            <b>Ana Živković</b> contributed to the Croatian adaptation.
          </li>
          <li>
            <b>Prof. Dr. Lidija Beko</b> led the Montenegrin localization.
          </li>
        </ul>
        <p>Thank you all.</p>

        <h3>Data attributions</h3>
        <p>
          The COPA task in both the Serbian (COPA-SR latin) and Croatian
          (COPA-HR) tracks of BalkanBench is taken from work by CLASSLA and
          collaborators. Please cite the original sources when using or
          building on these splits.
        </p>
        <p><b>COPA-SR (Serbian, latin)</b></p>
        <pre className="content-code">{`@misc{11356/1708,
  title     = {Choice of plausible alternatives dataset in Serbian {COPA}-{SR}},
  author    = {Ljube{\\v s}i{\\'c}, Nikola and Starovi{\\'c}, Mirjana and
               Kuzman, Taja and Samard{\\v z}i{\\'c}, Tanja},
  url       = {http://hdl.handle.net/11356/1708},
  note      = {Slovenian language resource repository {CLARIN}.{SI}},
  copyright = {Creative Commons - Attribution-{ShareAlike} 4.0 International
               ({CC} {BY}-{SA} 4.0)},
  issn      = {2820-4042},
  year      = {2022}
}`}</pre>
        <p><b>COPA-HR (Croatian)</b></p>
        <pre className="content-code">{`@article{DBLP:journals/corr/abs-2104-09243,
  author        = {Nikola Ljube{\\v s}i{\\'c} and Davor Lauc},
  title         = {BERTi{\\'c} - The Transformer Language Model for Bosnian,
                   Croatian, Montenegrin and Serbian},
  journal       = {CoRR},
  volume        = {abs/2104.09243},
  year          = {2021},
  url           = {https://arxiv.org/abs/2104.09243},
  archivePrefix = {arXiv}
}`}</pre>

        <h3>Contact, sponsorship, and future collaboration</h3>
        <p>
          This journey has been carried out by us at{' '}
          <a href="https://recrewty.com" target="_blank" rel="noopener noreferrer">
            <b>Recrewty</b>
          </a>
          . We will gladly sponsor compute for the next evaluations and
          submissions. If you are willing to contribute to this project as a
          sponsor, have ideas how to improve the benchmark, or would like to
          contribute in any other way, do not hesitate to reach out via any
          channel you find suitable.
        </p>
        <p>
          Email:{' '}
          <a href="mailto:balkanbench@recrewty.com">balkanbench@recrewty.com</a>
        </p>
      </section>
    </div>
  )
}
