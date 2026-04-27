export default function Topbar() {
  return (
    <div className="topbar">
      <div className="tickers">
        <span className="ticker-item"><span className="dot" /><b>STATUS</b> · v0.1 LIVE</span>
        <span className="ticker-sep">//</span>
        <span className="ticker-item">RELEASED 2026-04-27</span>
        <span className="ticker-sep">//</span>
        <span className="ticker-item">SR · HR · MNE · BS</span>
        <span className="ticker-sep">//</span>
        <span className="ticker-item">MIT LICENSED</span>
        <span style={{ marginLeft: 'auto' }} className="ticker-item">
          SPONSORED BY <b style={{ marginLeft: 4 }}>RECREWTY</b>
        </span>
      </div>
    </div>
  )
}
