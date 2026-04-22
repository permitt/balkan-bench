# BalkanBench v0.1 - Plan 6: Frontend

**Goal:** Ship a 4-route React app in `frontend/`: `/` (existing landing),
`/leaderboard` (renders `benchmark_results.json`), `/about`, `/submit`.
Vercel deploys from `frontend/` with SPA routing. Launch leaderboard
seeded with the nine v0.1 rows.

**Architecture:**
- `react-router-dom` for routes; `createBrowserRouter` + `RouterProvider`.
- Pages under `frontend/src/pages/`; existing App.jsx content becomes `pages/Home.jsx`.
- `Leaderboard.jsx` fetches `/leaderboards/superglue-serbian/benchmark_results.json` at runtime and renders the table.
- Editorial brutalist styling continues: one shared stylesheet, shared `Topbar`, `Marquee`, `Footer` components.
- `frontend/vercel.json` rewrites every unmatched path to `index.html` so deep links work.

**Branch:** `feature/code-for-eval`.

---

## Task 1: Add react-router-dom, restructure App

- `npm install react-router-dom` in `frontend/`.
- Move existing landing content into `pages/Home.jsx`; replace `App.jsx` with a `RouterProvider`.
- Split reusable bits into `components/{Topbar,Marquee,Footer}.jsx`.
- `npm run build` must still succeed.
- Commit: `feat(frontend): add react-router, restructure App into pages/`.

## Task 2: Seed `benchmark_results.json` with the nine launch rows

- `frontend/public/leaderboards/superglue-serbian/benchmark_results.json` with the user's published table (SuperGLUE-SR, 9 models, ModernBERTic small as 5/6 partial).
- Conforms to `eval/schemas/leaderboard_export.json`.
- Commit: `feat(frontend): seed benchmark_results.json with the v0.1 launch rows`.

## Task 3: `/leaderboard` page

- Fetches the JSON, renders a sortable table with 6 task columns + Avg + Params + Throughput (null until Plan 5 artifacts land).
- Partial rows show `(5/6) partial` badge and no rank number.
- Styled to match the landing page (same monospace tickers, same colour variables).
- Commit: `feat(frontend): /leaderboard page rendering benchmark_results.json`.

## Task 4: `/about` + `/submit` pages

- `/about` summarises methodology + language coverage + sponsor block, links to the spec and governance docs.
- `/submit` walks the 3-step submission flow and links to the GitHub issue template.
- Commit: `feat(frontend): /about and /submit pages`.

## Task 5: Vercel config + production build smoke

- `frontend/vercel.json` sets `framework: vite`, `rewrites: /* -> /index.html`, `outputDirectory: dist`.
- `npm run build` smoke.
- Commit: `build(frontend): vercel SPA rewrites + build smoke`.

## Task 6: `frontend/README.md`

- Overview, local dev (`npm install && npm run dev`), build, Vercel deploy notes, where `benchmark_results.json` lives and how it gets refreshed.
- Commit: `docs(frontend): dev + deploy walkthrough`.

## End of Plan 6

Success state: `cd frontend && npm run dev` serves `/`, `/leaderboard`,
`/about`, `/submit`; `npm run build` succeeds; deep links work on Vercel
thanks to the rewrites.
