import { formatPlayerId } from '../lib/format'
import type { WavelengthObservation } from '../lib/types'

interface WavelengthTranscriptProps {
  observation: WavelengthObservation
}

export default function WavelengthTranscript({ observation }: WavelengthTranscriptProps) {
  return (
    <section className="panel card wavelength-transcript">
      <h3>Wavelength Transcript</h3>

      {observation.pending_exchange ? (
        <div className="wavelength-pending">
          <strong>Current prompt:</strong>{' '}
          {formatPlayerId(observation.pending_exchange.guesser_id)} asked{' '}
          <em>{observation.pending_exchange.category}</em>
          {observation.pending_exchange.answer ? ` -> ${observation.pending_exchange.answer}` : ''}
        </div>
      ) : null}

      <ul className="wavelength-history">
        {observation.history.map((entry, index) => (
          <li key={`${entry.round}-${entry.guesser_id}-${index}`}>
            <div>
              <strong>Round {entry.round}</strong> - {formatPlayerId(entry.guesser_id)}
            </div>
            <div>Category: {entry.category}</div>
            <div>Answer: {entry.answer ?? '-'}</div>
            <div>Guess: {entry.guess ?? '-'}</div>
          </li>
        ))}
      </ul>

      {observation.history.length === 0 ? <p className="muted">No exchanges yet.</p> : null}
    </section>
  )
}
