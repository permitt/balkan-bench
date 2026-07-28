const TASK_LABELS = {
  boolq: 'BoolQ',
  cb: 'CB',
  copa: 'COPA',
  rte: 'RTE',
  multirc: 'MultiRC',
  wsc: 'WSC',
  arc_challenge: 'ARC-C',
  arc_easy: 'ARC-E',
  hellaswag: 'HellaSwag',
  nq_open: 'NQ',
  openbookqa: 'OBQA',
  piqa: 'PIQA',
  triviaqa: 'TriviaQA',
  winogrande: 'WinoG',
}

const TASK_DESCRIPTIONS = {
  boolq: 'Boolean questions over a short passage',
  cb: 'Three-way textual entailment (entailment / contradiction / neutral)',
  copa: 'Choice of plausible alternatives (cause / effect)',
  rte: 'Binary textual entailment',
  multirc: 'Multi-sentence reading comprehension (grouped F1 + exact match)',
  wsc: 'Winograd Schema coreference reformulated as binary classification',
  arc_challenge: 'Grade-school science questions, challenge split, Serbian adaptation',
  arc_easy: 'Grade-school science questions, easy split, Serbian adaptation',
  hellaswag: 'Commonsense sentence completion, Serbian adaptation',
  nq_open: 'Open-domain question answering, Serbian adaptation',
  openbookqa: 'Open-book science QA over elementary-level facts, Serbian adaptation',
  piqa: 'Physical commonsense reasoning between two candidate solutions, Serbian adaptation',
  triviaqa: 'Trivia question answering, Serbian adaptation',
  winogrande: 'Winograd-style pronoun resolution, Serbian adaptation',
}

const LANGUAGES = {
  sr:  { flag: '🇷🇸', name: 'Serbian',     nativeName: 'Srpski' },
  hr:  { flag: '🇭🇷', name: 'Croatian',    nativeName: 'Hrvatski' },
  mne: { flag: '🇲🇪', name: 'Montenegrin', nativeName: 'Crnogorski' },
  bs:  { flag: '🇧🇦', name: 'Bosnian',     nativeName: 'Bosanski' },
}

const BENCHMARKS = {
  superglue: {
    label: 'SuperGLUE',
    tagline: 'Encoder NLU · 6 ranked tasks',
    description: 'Encoder-fine-tune NLU, 6 ranked tasks + 2 diagnostics.',
    available: true,
    availableIn: null,
  },
  sle: {
    label: 'Serbian-LLM-Eval',
    tagline: 'Generative few-shot',
    description: 'Generative few-shot eval (Aleksa Gordić) - ARC, HellaSwag, PIQA, BoolQ, Winogrande, etc.',
    available: true,
    availableIn: null,
  },
  mteb_bcms: {
    label: 'MTEB-BCMS',
    tagline: 'Embeddings · 4 tasks',
    description: 'Massive Text Embedding Benchmark, BCMS adaptation.',
    available: false,
    availableIn: 'v1.2',
  },
  llm_arena: {
    label: 'LLM Arena',
    tagline: 'Human-judged Elo',
    description: 'Head-to-head human preference ratings across BCMS LLMs.',
    available: false,
    availableIn: 'v1.2',
  },
}

// Discoverable leaderboards. When a new (benchmark, language) pair publishes
// its benchmark_results.json, flip `available: true`; no other code changes.
const LEADERBOARDS = [
  { benchmark: 'superglue', language: 'sr',  path: 'superglue-sr',  available: true,  availableIn: null   },
  { benchmark: 'superglue', language: 'hr',  path: 'superglue-hr',  available: true,  availableIn: null   },
  { benchmark: 'superglue', language: 'mne', path: 'superglue-mne', available: true,  availableIn: null   },
  { benchmark: 'superglue', language: 'bs',  path: 'superglue-bs',  available: false, availableIn: 'v1.1' },
  { benchmark: 'sle',       language: 'sr',  path: 'sle-sr',        available: true,  availableIn: null   },
]

export { TASK_LABELS, TASK_DESCRIPTIONS, LANGUAGES, BENCHMARKS, LEADERBOARDS }

// Maintained release constants, updated when new boards publish.
// items = 67,313 SuperGLUE (train+val+test, SR) + 42,019 SLE test items.
// modelCount = unique model_ids across all published boards.
export const FACTS = { items: 109332, languageCount: 3, modelCount: 19 }

export function formatCell(cell) {
  if (cell === null || cell === undefined) return { main: '-', stdev: null }
  // Artifacts store sklearn-native 0-1 values; render as 0-100 at display
  // time so on-disk artifacts stay canonical.
  const mean = (Number(cell.mean) * 100).toFixed(2)
  const stdev = cell.stdev === undefined ? null : (Number(cell.stdev) * 100).toFixed(2)
  return { main: mean, stdev }
}

export function formatAvg(row) {
  return (row.avg * 100).toFixed(2)
}

// Display-only: sle exports prefix run names with "sle-"; the raw row.model
// stays the row key and the ?model= deep-link value.
export function displayModelName(model) {
  return model.startsWith('sle-') ? model.slice(4) : model
}

export function sortValue(row, rankBy) {
  if (rankBy === 'avg') return row.avg ?? null
  const cell = row.results[rankBy]
  return cell ? cell.mean : null
}

export function sortRows(rows, rankBy) {
  const sorted = [...rows]
  sorted.sort((a, b) => {
    const av = sortValue(a, rankBy)
    const bv = sortValue(b, rankBy)
    if (av === null && bv === null) return 0
    if (av === null) return 1
    if (bv === null) return -1
    return bv - av
  })
  return sorted
}

export function resolveBoard(bench, lang) {
  return (
    LEADERBOARDS.find(l => l.benchmark === bench && l.language === lang && l.available) ??
    LEADERBOARDS.find(l => l.available) ??
    LEADERBOARDS[0]
  )
}

export function boardEntriesFor(benchmark) {
  return LEADERBOARDS.filter(l => l.benchmark === benchmark)
}

export function boardUrl(path) {
  return `/leaderboards/${path}/benchmark_results.json`
}
