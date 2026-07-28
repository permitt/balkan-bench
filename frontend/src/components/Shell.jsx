import { Link, NavLink, Outlet } from 'react-router-dom'
import { MotionConfig } from 'motion/react'
import { useTheme } from '../lib/useTheme.js'
import '../styles/shell.css'

const CYCLE = { auto: 'light', light: 'dark', dark: 'auto' }
const THEME_ICON = { auto: 'A', light: '☀', dark: '☽' }

export default function Shell() {
  const { theme, setTheme } = useTheme()

  return (
    <MotionConfig reducedMotion="user">
      <header className="shell-nav">
        <div className="shell-nav-inner container">
          <Link to="/" className="shell-brand">BalkanBench</Link>
          <nav className="shell-links">
            <NavLink to="/leaderboard">Leaderboard</NavLink>
            <NavLink to="/about">About</NavLink>
            <NavLink to="/submit">Submit</NavLink>
            <a href="https://github.com/permitt/balkan-bench" target="_blank" rel="noopener noreferrer">GitHub</a>
          </nav>
          <button
            type="button"
            className="shell-theme"
            aria-label={`Theme: ${theme}`}
            onClick={() => setTheme(CYCLE[theme])}
          >
            {THEME_ICON[theme]}
          </button>
        </div>
      </header>
      <main className="shell-main">
        <Outlet />
      </main>
      <footer className="shell-footer">
        <div className="container shell-footer-inner">
          <span>Compute sponsored by <a href="https://recrewty.com" target="_blank" rel="noopener noreferrer">Recrewty</a></span>
          <span className="shell-footer-meta">(c) 2026 BalkanBench - MIT License</span>
        </div>
      </footer>
    </MotionConfig>
  )
}
