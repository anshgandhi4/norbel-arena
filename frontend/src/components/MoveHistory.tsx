import type { MatchEvent } from '../lib/types'

interface MoveHistoryProps {
  events: MatchEvent[]
}

export default function MoveHistory({ events }: MoveHistoryProps) {
  const recent = events.slice(-20).reverse()

  return (
    <section className="panel card">
      <h3>Move History</h3>
      <ul className="history-list">
        {recent.map((event, idx) => {
          const summary = String(event.payload?.summary ?? '')
          return (
            <li key={`${event.timestamp_ms}-${idx}`}>
              <div className="history-top">
                <span>{event.event_type}</span>
                <span>T{event.turn}</span>
              </div>
              <div className="history-body">{summary || JSON.stringify(event.payload)}</div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
