import type { AgentReportCard, ReportCards } from '../lib/types'

interface ReportCardPanelProps {
  reportCards: ReportCards | null
}

function roleFromLabel(label: string): 'SPYMASTER' | 'OPERATIVE' | null {
  if (label.endsWith('[SPYMASTER]')) {
    return 'SPYMASTER'
  }
  if (label.endsWith('[OPERATIVE]')) {
    return 'OPERATIVE'
  }
  return null
}

function shouldIncludeLabel(gameName: string, label: string): boolean {
  const role = roleFromLabel(label)
  if (gameName.endsWith(':operator_eval') && role === 'SPYMASTER') {
    return false
  }
  if (gameName.endsWith(':spymaster_eval') && role === 'OPERATIVE') {
    return false
  }
  return true
}

function rowsForGame(
  gameName: string,
  gameData: Record<string, AgentReportCard>
): Array<{ label: string; stats: AgentReportCard }> {
  return Object.entries(gameData)
    .filter(([label]) => shouldIncludeLabel(gameName, label))
    .map(([label, stats]) => ({ label, stats }))
    .sort((a, b) => b.stats.matches - a.stats.matches)
}

export default function ReportCardPanel({ reportCards }: ReportCardPanelProps) {
  if (!reportCards) {
    return (
      <section className="panel card">
        <h3>Arena Leaderboards</h3>
        <p className="muted">No results yet.</p>
      </section>
    )
  }

  const gameEntries = Object.entries(reportCards.games)
  return (
    <section className="panel card">
      <h3>Arena Leaderboards</h3>
      <p className="muted">Updated {new Date(reportCards.updated_at).toLocaleString()}</p>
      {gameEntries.length === 0 ? <p className="muted">No completed matches recorded yet.</p> : null}
      {gameEntries.map(([gameName, gameData]) => {
        const rows = rowsForGame(gameName, gameData)
        return (
          <div key={gameName} className="report-block">
            <h4>{gameName}</h4>
            {rows.length === 0 ? (
              <p className="muted">No rows for this evaluation mode yet.</p>
            ) : (
              <div className="report-table-wrap">
                <table className="report-table">
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Matches</th>
                      <th>W-L-D</th>
                      <th>Win %</th>
                      <th>Avg Turns</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ label, stats }) => (
                      <tr key={label}>
                        <td>{label}</td>
                        <td>{stats.matches}</td>
                        <td>
                          {stats.wins}-{stats.losses}-{stats.draws}
                        </td>
                        <td>{(stats.win_rate * 100).toFixed(1)}%</td>
                        <td>{stats.avg_turns.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )
      })}
    </section>
  )
}
