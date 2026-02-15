import { formatPlayerId, replacePlayerIdsInText } from '../lib/format'
import type { MatchEvent } from '../lib/types'

interface MoveHistoryProps {
  events: MatchEvent[]
  selectedTurn: number
  onSelectTurn: (turn: number) => void
}

export default function MoveHistory({ events, selectedTurn, onSelectTurn }: MoveHistoryProps) {
  const recent = events.slice().reverse().slice(0, 50)

  return (
    <section className="panel card">
      <h3>Move History</h3>
      <ul className="history-list">
        {recent.map((event, idx) => {
          const summary = replacePlayerIdsInText(String(event.payload?.summary ?? ''))
          const payload = { ...event.payload }
          delete payload.prompt_context
          delete payload.model_response
          delete payload.model_response_history
          if (typeof payload.player_id === 'string') {
            payload.player_id = formatPlayerId(payload.player_id)
          }
          const fallbackJson = replacePlayerIdsInText(JSON.stringify(payload, null, 2))
          const selectable = Number.isFinite(event.turn)
          const active = selectable && event.turn === selectedTurn
          return (
            <li key={`${event.timestamp_ms}-${idx}`} className={active ? 'active' : ''}>
              <div className="history-top">
                <span>{event.event_type}</span>
                <button
                  className="link-button"
                  disabled={!selectable}
                  onClick={() => onSelectTurn(event.turn)}
                >
                  T{event.turn}
                </button>
              </div>
              <div className="history-body">{summary || fallbackJson}</div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
