import type { ReactNode } from 'react'
import norbelLogo from '../../norbel_logo.png'
import norbelLogoDark from '../../norbel_logo_dark.png'

export type Theme = 'dark' | 'light'

interface HeaderLink {
  href: string
  label: string
}

interface AppHeaderProps {
  centerLabel?: string
  leftLinks?: HeaderLink[]
  rightControls?: ReactNode
  theme: Theme
  onToggleTheme: () => void
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="4.2" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 2.5v2.2M12 19.3v2.2M21.5 12h-2.2M4.7 12H2.5M18.7 5.3l-1.6 1.6M6.9 17.1l-1.6 1.6M18.7 18.7l-1.6-1.6M6.9 6.9L5.3 5.3"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M16.8 14.4a7.4 7.4 0 0 1-7.2-9.5 8.8 8.8 0 1 0 9.5 9.5 7.2 7.2 0 0 1-2.3 0z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

export default function AppHeader({ centerLabel, leftLinks, rightControls, theme, onToggleTheme }: AppHeaderProps) {
  const toggleLabel = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
  const logoSrc = theme === 'light' ? norbelLogoDark : norbelLogo

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="app-header-brand">
          <a className="app-title-link" href="/">
            <img className="app-logo" src={logoSrc} alt="Norbel Arena" />
          </a>
        </div>

        {centerLabel || (leftLinks && leftLinks.length > 0) ? (
          <div className="app-header-center">
            {centerLabel ? (
              <span className="app-header-center-label">{centerLabel}</span>
            ) : (
              <nav className="app-header-links" aria-label="Page links">
                {leftLinks?.map((link) => (
                  <a key={`${link.href}-${link.label}`} className="header-link" href={link.href}>
                    {link.label}
                  </a>
                ))}
              </nav>
            )}
          </div>
        ) : null}

        <div className="app-header-actions">
          {rightControls}
          <a
            className="pro-header-button"
            href="https://www.youtube.com/watch?v=lIsTm0lGzY4"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <rect x="3.5" y="10" width="17" height="10" rx="2" fill="none" stroke="currentColor" strokeWidth="2" />
              <path d="M12 10v10M4 13h16" fill="none" stroke="currentColor" strokeWidth="2" />
              <path
                d="M12 10H8.7A2.7 2.7 0 1 1 8.7 4.6C10.5 4.6 12 6.1 12 7.9V10ZM12 10h3.3A2.7 2.7 0 1 0 15.3 4.6C13.5 4.6 12 6.1 12 7.9V10Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <span>Pro</span>
          </a>
          <button
            type="button"
            className="theme-toggle icon-toggle"
            onClick={onToggleTheme}
            aria-label={toggleLabel}
            title={toggleLabel}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </div>
    </header>
  )
}
