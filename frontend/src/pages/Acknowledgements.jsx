import Topbar from '../components/Topbar.jsx'
import Nav from '../components/Nav.jsx'
import Footer from '../components/Footer.jsx'
import '../styles/content.css'

export default function Acknowledgements() {
  return (
    <>
      <Topbar />
      <Nav />
      <section className="content-wrap">
        <div className="eyebrow">
          <span className="chip">ACKNOWLEDGEMENTS</span>
          <span>CONTRIBUTORS · DATA · SPONSORS</span>
        </div>
        <h1 className="content-title">
          Built with <span className="stroke">many hands</span>.
        </h1>
        <p className="content-lede">
          BalkanBench exists because annotators, linguists, and researchers
          across the region donated their time and expertise. This page
          records who they are, what they contributed, and where the data
          comes from.
        </p>

        <h2>Contributing</h2>
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

        <h2>Acknowledgements</h2>
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

        <h2>Data attributions</h2>
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

        <h2>Contact, sponsorship, and future collaboration</h2>
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
      <Footer />
    </>
  )
}
