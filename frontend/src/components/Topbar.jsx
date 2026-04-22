export default function Topbar() {
  return (
    <div className="topbar">
      <div className="tickers">
        <span className="ticker-item"><span className="dot" /><b>STATUS</b> · building v1.2</span>
        <span className="ticker-sep">//</span>
        <span className="ticker-item">LAUNCHING Q2 2026</span>
        <span className="ticker-sep">//</span>
        <span className="ticker-item">SR · ME · HR · BS</span>
        <span className="ticker-sep">//</span>
        <span className="ticker-item">MIT LICENSED</span>
        <span style={{ marginLeft: 'auto' }} className="ticker-item">
          SPONSORED BY <b style={{ marginLeft: 4 }}>RECREWTY</b>
        </span>
      </div>
    </div>
  )
}
