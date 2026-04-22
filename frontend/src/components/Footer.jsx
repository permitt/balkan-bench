export default function Footer() {
  return (
    <div className="bottom">
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <span className="sponsor-k">§ SPONSORED BY</span>
        <a href="https://recrewty.com" target="_blank" rel="noopener noreferrer" className="recrewty">
          <img src="/recrewty-logo.png" alt="Recrewty" />
        </a>
      </div>
      <div className="bottom-meta">
        <span>© 2026 BALKANBENCH</span>
        <span className="sep">/</span>
        <span>MIT LICENSE</span>
        <span className="sep">/</span>
        <span>V1.2 · PRE-RELEASE</span>
      </div>
    </div>
  )
}
