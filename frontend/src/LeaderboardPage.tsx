import { useEffect, useMemo, useState } from 'react'
import AppHeader, { type Theme } from './components/AppHeader'
import { fetchReportCards } from './lib/api'
import type { ReportCards } from './lib/types'

type LeaderboardRole = 'SPYMASTER' | 'OPERATIVE' | 'GUESSER'

interface AggregatedRoleRow {
  agent: string
  matches: number
  wins: number
  losses: number
  draws: number
  totalTurns: number
  avgTurns: number
  winRate: number
  elo: number | null
  eloSampleMatches: number
  lastPlayed: string | null
}

interface CodenamesRow {
  agent: string
  overallElo: number
  spymasterElo: number | null
  operativeElo: number | null
  matches: number
  spymasterMatches: number
  operativeMatches: number
  wins: number
  losses: number
  draws: number
  avgTurns: number
  lastPlayed: string | null
}

const THEME_STORAGE_KEY = 'treehacks-ui-theme'
const ELO_BASE_RATING = 1000
const CODENAMES_SPYMASTER_WEIGHT = 0.65
const CODENAMES_OPERATIVE_WEIGHT = 0.35

function resolveInitialTheme(): Theme {
  if (typeof window === 'undefined') {
    return 'dark'
  }
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
  return storedTheme === 'light' ? 'light' : 'dark'
}

function roleFromLabel(label: string): LeaderboardRole | null {
  if (label.endsWith('[SPYMASTER]')) {
    return 'SPYMASTER'
  }
  if (label.endsWith('[OPERATIVE]')) {
    return 'OPERATIVE'
  }
  if (label.endsWith('[GUESSER]')) {
    return 'GUESSER'
  }
  return null
}

function baseAgentLabel(label: string): string {
  return label.replace(/\s*\[(SPYMASTER|OPERATIVE|GUESSER)\]\s*$/, '').trim()
}

function pickMostRecent(current: string | null, incoming: string | null): string | null {
  if (!incoming) {
    return current
  }
  if (!current) {
    return incoming
  }
  const currentTime = Date.parse(current)
  const incomingTime = Date.parse(incoming)
  if (Number.isNaN(currentTime)) {
    return incoming
  }
  if (Number.isNaN(incomingTime)) {
    return current
  }
  return incomingTime > currentTime ? incoming : current
}

function aggregateRowsByRole(
  reportCards: ReportCards | null,
  role: LeaderboardRole,
  gamePrefix: string | null = null
): AggregatedRoleRow[] {
  if (!reportCards) {
    return []
  }

  const merged = new Map<string, AggregatedRoleRow>()

  for (const [gameName, gameData] of Object.entries(reportCards.games)) {
    if (gamePrefix && !gameName.startsWith(gamePrefix)) {
      continue
    }

    for (const [label, stats] of Object.entries(gameData)) {
      if (roleFromLabel(label) !== role) {
        continue
      }

      const agent = baseAgentLabel(label)
      const existing = merged.get(agent)
      const matches = Number(stats.matches) || 0
      const totalTurns = Number(stats.total_turns) || Math.round(Number(stats.avg_turns) * matches)
      const incomingElo = Number.isFinite(Number(stats.elo)) ? Number(stats.elo) : null

      if (!existing) {
        merged.set(agent, {
          agent,
          matches,
          wins: Number(stats.wins) || 0,
          losses: Number(stats.losses) || 0,
          draws: Number(stats.draws) || 0,
          totalTurns,
          avgTurns: 0,
          winRate: 0,
          elo: incomingElo,
          eloSampleMatches: incomingElo === null ? 0 : Math.max(1, matches),
          lastPlayed: stats.last_played ?? null
        })
        continue
      }

      existing.matches += matches
      existing.wins += Number(stats.wins) || 0
      existing.losses += Number(stats.losses) || 0
      existing.draws += Number(stats.draws) || 0
      existing.totalTurns += totalTurns
      existing.lastPlayed = pickMostRecent(existing.lastPlayed, stats.last_played ?? null)
      if (incomingElo !== null) {
        const incomingWeight = Math.max(1, matches)
        if (existing.elo === null || existing.eloSampleMatches <= 0) {
          existing.elo = incomingElo
          existing.eloSampleMatches = incomingWeight
        } else {
          const totalWeight = existing.eloSampleMatches + incomingWeight
          existing.elo =
            (existing.elo * existing.eloSampleMatches + incomingElo * incomingWeight) / totalWeight
          existing.eloSampleMatches = totalWeight
        }
      }
    }
  }

  const rows = Array.from(merged.values())
  for (const row of rows) {
    row.winRate = row.matches > 0 ? row.wins / row.matches : 0
    row.avgTurns = row.matches > 0 ? row.totalTurns / row.matches : 0
  }
  return rows
}

function aggregateCodenamesRows(reportCards: ReportCards | null): CodenamesRow[] {
  const spymasterRows = aggregateRowsByRole(reportCards, 'SPYMASTER', 'codenames:')
  const operativeRows = aggregateRowsByRole(reportCards, 'OPERATIVE', 'codenames:')

  const spymasterByAgent = new Map(spymasterRows.map((row) => [row.agent, row]))
  const operativeByAgent = new Map(operativeRows.map((row) => [row.agent, row]))
  const allAgents = new Set([...spymasterByAgent.keys(), ...operativeByAgent.keys()])

  const rows: CodenamesRow[] = []
  for (const agent of allAgents) {
    const spymaster = spymasterByAgent.get(agent)
    const operative = operativeByAgent.get(agent)

    const spymasterElo = spymaster?.elo ?? null
    const operativeElo = operative?.elo ?? null
    const effectiveSpymasterElo = spymasterElo ?? ELO_BASE_RATING
    const effectiveOperativeElo = operativeElo ?? ELO_BASE_RATING
    const overallElo =
      effectiveSpymasterElo * CODENAMES_SPYMASTER_WEIGHT +
      effectiveOperativeElo * CODENAMES_OPERATIVE_WEIGHT

    const matches = (spymaster?.matches ?? 0) + (operative?.matches ?? 0)
    const totalTurns = (spymaster?.totalTurns ?? 0) + (operative?.totalTurns ?? 0)
    rows.push({
      agent,
      overallElo,
      spymasterElo,
      operativeElo,
      matches,
      spymasterMatches: spymaster?.matches ?? 0,
      operativeMatches: operative?.matches ?? 0,
      wins: (spymaster?.wins ?? 0) + (operative?.wins ?? 0),
      losses: (spymaster?.losses ?? 0) + (operative?.losses ?? 0),
      draws: (spymaster?.draws ?? 0) + (operative?.draws ?? 0),
      avgTurns: matches > 0 ? totalTurns / matches : 0,
      lastPlayed: pickMostRecent(spymaster?.lastPlayed ?? null, operative?.lastPlayed ?? null)
    })
  }

  return rows.sort((a, b) => {
    if (b.overallElo !== a.overallElo) {
      return b.overallElo - a.overallElo
    }
    if (b.wins !== a.wins) {
      return b.wins - a.wins
    }
    if (b.matches !== a.matches) {
      return b.matches - a.matches
    }
    return a.agent.localeCompare(b.agent)
  })
}

function aggregateWavelengthRows(reportCards: ReportCards | null): AggregatedRoleRow[] {
  const rows = aggregateRowsByRole(reportCards, 'GUESSER', 'wavelength:')
  return rows.sort((a, b) => {
    const aElo = a.elo ?? Number.NEGATIVE_INFINITY
    const bElo = b.elo ?? Number.NEGATIVE_INFINITY
    if (bElo !== aElo) {
      return bElo - aElo
    }
    if (b.wins !== a.wins) {
      return b.wins - a.wins
    }
    if (b.matches !== a.matches) {
      return b.matches - a.matches
    }
    return a.agent.localeCompare(b.agent)
  })
}

function formatRelativeTime(isoTimestamp: string | null): string {
  if (!isoTimestamp) {
    return 'No updates yet'
  }

  const timestamp = Date.parse(isoTimestamp)
  if (Number.isNaN(timestamp)) {
    return 'Updated recently'
  }

  const diffMs = Date.now() - timestamp
  const diffSeconds = Math.floor(Math.abs(diffMs) / 1000)

  if (diffSeconds < 60) {
    return 'Updated just now'
  }

  const units: Array<{ unit: string; seconds: number }> = [
    { unit: 'day', seconds: 86400 },
    { unit: 'hour', seconds: 3600 },
    { unit: 'minute', seconds: 60 }
  ]

  for (const { unit, seconds } of units) {
    if (diffSeconds >= seconds) {
      const value = Math.floor(diffSeconds / seconds)
      const suffix = value === 1 ? '' : 's'
      return diffMs >= 0
        ? `Updated ${value} ${unit}${suffix} ago`
        : `Updated in ${value} ${unit}${suffix}`
    }
  }

  return 'Updated recently'
}

function formatDateTime(isoTimestamp: string | null): string {
  if (!isoTimestamp) {
    return '-'
  }

  const timestamp = Date.parse(isoTimestamp)
  if (Number.isNaN(timestamp)) {
    return '-'
  }

  return new Date(timestamp).toLocaleDateString()
}

function CodenamesLeaderboardTable(props: {
  rows: CodenamesRow[]
  updatedAt: string | null
  emptyLabel: string
}) {
  const { rows, updatedAt, emptyLabel } = props

  return (
    <article className="leaderboard-card">
      <header className="leaderboard-card-head">
        <h3>Codenames</h3>
        <p className="muted">{formatRelativeTime(updatedAt)}</p>
      </header>
      {rows.length === 0 ? (
        <p className="muted">{emptyLabel}</p>
      ) : (
        <div className="leaderboard-table-wrap">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Agent</th>
                <th>Overall Elo</th>
                <th>Spy Elo</th>
                <th>Op Elo</th>
                <th>Matches (S/O)</th>
                <th>W-L-D</th>
                <th>Avg Turns</th>
                <th>Last Played</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={row.agent}>
                  <td>{index + 1}</td>
                  <td>{row.agent}</td>
                  <td>{row.overallElo.toFixed(1)}</td>
                  <td>{row.spymasterElo !== null ? row.spymasterElo.toFixed(1) : '-'}</td>
                  <td>{row.operativeElo !== null ? row.operativeElo.toFixed(1) : '-'}</td>
                  <td>
                    {row.matches} ({row.spymasterMatches}/{row.operativeMatches})
                  </td>
                  <td>
                    {row.wins}-{row.losses}-{row.draws}
                  </td>
                  <td>{row.avgTurns.toFixed(1)}</td>
                  <td>{formatDateTime(row.lastPlayed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  )
}

function WavelengthLeaderboardTable(props: {
  rows: AggregatedRoleRow[]
  updatedAt: string | null
  emptyLabel: string
}) {
  const { rows, updatedAt, emptyLabel } = props

  return (
    <article className="leaderboard-card">
      <header className="leaderboard-card-head">
        <h3>Wavelength</h3>
        <p className="muted">{formatRelativeTime(updatedAt)}</p>
      </header>
      {rows.length === 0 ? (
        <p className="muted">{emptyLabel}</p>
      ) : (
        <div className="leaderboard-table-wrap">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Agent</th>
                <th>Elo</th>
                <th>Matches</th>
                <th>W-L-D</th>
                <th>Avg Turns</th>
                <th>Last Played</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={row.agent}>
                  <td>{index + 1}</td>
                  <td>{row.agent}</td>
                  <td>{row.elo !== null ? row.elo.toFixed(1) : '-'}</td>
                  <td>{row.matches}</td>
                  <td>
                    {row.wins}-{row.losses}-{row.draws}
                  </td>
                  <td>{row.avgTurns.toFixed(1)}</td>
                  <td>{formatDateTime(row.lastPlayed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  )
}

export default function LeaderboardPage() {
  const [theme, setTheme] = useState<Theme>(() => resolveInitialTheme())
  const [reportCards, setReportCards] = useState<ReportCards | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    document.documentElement.style.colorScheme = theme
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const payload = await fetchReportCards()
        if (!cancelled) {
          setReportCards(payload)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  const codenamesRows = useMemo(() => aggregateCodenamesRows(reportCards), [reportCards])
  const wavelengthRows = useMemo(() => aggregateWavelengthRows(reportCards), [reportCards])

  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))

  return (
    <main className="app leaderboard-page">
      <AppHeader centerLabel="Leaderboards" theme={theme} onToggleTheme={toggleTheme} />

      {loading ? <p className="muted leaderboard-status">Loading leaderboards...</p> : null}
      {error ? <div className="error-banner">{error}</div> : null}

      <section className="leaderboard-grid">
        <CodenamesLeaderboardTable
          rows={codenamesRows}
          updatedAt={reportCards?.updated_at ?? null}
          emptyLabel="No Codenames stats recorded yet."
        />
        <WavelengthLeaderboardTable
          rows={wavelengthRows}
          updatedAt={reportCards?.updated_at ?? null}
          emptyLabel="No Wavelength stats recorded yet."
        />
      </section>
    </main>
  )
}
