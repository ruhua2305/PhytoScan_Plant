export default function Header() {
  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Logo */}
        <a href="/" className="logo" aria-label="LeafScan home">
          <LeafLogoIcon />
          <span className="logo-text">
            Leaf<span className="logo-accent">Scan</span>
          </span>
        </a>

        {/* Nav */}
        <nav className="header-nav" aria-label="Main navigation">
          <a
            href="https://github.com"
            className="nav-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          <a href="#how-it-works" className="nav-link">
            How it works
          </a>
          <a href="#about" className="nav-link nav-cta">
            About the model
          </a>
        </nav>
      </div>
    </header>
  );
}

function LeafLogoIcon() {
  return (
    <svg
      className="logo-icon"
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M16 4C8 4 4 11 4 17C4 23 9 28 16 28C23 28 28 23 28 17C28 11 24 4 16 4Z"
        fill="currentColor"
        opacity="0.2"
      />
      <path
        d="M16 5C9 5 5 11.5 5 17C5 22.5 10 27 16 27"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M16 5C23 5 27 11.5 27 17C27 22.5 22 27 16 27"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
      <path d="M16 27V7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path
        d="M16 14L11 10M16 19L10 16M16 23L11 21"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        opacity="0.8"
      />
      <path
        d="M16 14L21 10M16 19L22 16M16 23L21 21"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        opacity="0.8"
      />
    </svg>
  );
}
