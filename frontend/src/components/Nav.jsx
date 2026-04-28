import { Link, NavLink } from 'react-router-dom'

export default function Nav() {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <Link to="/" className="mark">
          <span className="mark-glyph">B</span>
          <span>balkan<em>·</em>bench</span>
          <span className="mark-sub">BCMS EVAL SUITE</span>
        </Link>
        <div className="nav-links">
          <NavLink to="/leaderboard" end>Leaderboard</NavLink>
          <NavLink to="/about">About</NavLink>
          <NavLink to="/submit">Submit</NavLink>
          <a href="https://github.com/permitt/balkan-bench" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}
