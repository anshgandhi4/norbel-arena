import { useMemo, useState } from 'react'
import type { LegalMovesSpec, Observation } from '../lib/types'

interface ControlsPanelProps {
  observation: Observation
  legalMovesSpec: LegalMovesSpec
  isHumanTurn: boolean
  onGiveClue: (clue: string, count: number) => void
  onEndTurn: () => void
  onResign: () => void
}

export default function ControlsPanel({
  observation,
  legalMovesSpec,
  isHumanTurn,
  onGiveClue,
  onEndTurn,
  onResign
}: ControlsPanelProps) {
  const [clue, setClue] = useState('')
  const [count, setCount] = useState(1)

  const canGiveClue = useMemo(() => {
    const allowed = legalMovesSpec.allowed?.GiveClue
    return isHumanTurn && observation.phase === 'SPYMASTER_CLUE' && !!allowed
  }, [isHumanTurn, legalMovesSpec.allowed, observation.phase])

  const canEndTurn = useMemo(() => {
    return isHumanTurn && observation.phase === 'OPERATIVE_GUESSING' && legalMovesSpec.allowed?.EndTurn === true
  }, [isHumanTurn, legalMovesSpec.allowed, observation.phase])

  const canResign = useMemo(() => {
    return isHumanTurn && legalMovesSpec.allowed?.Resign !== false
  }, [isHumanTurn, legalMovesSpec.allowed])

  return (
    <section className="panel card">
      <h3>Controls</h3>

      {observation.phase === 'SPYMASTER_CLUE' ? (
        <div className="control-group">
          <label>
            Clue
            <input value={clue} onChange={(e) => setClue(e.target.value)} disabled={!canGiveClue} />
          </label>
          <label>
            Count
            <input
              type="number"
              min={0}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              disabled={!canGiveClue}
            />
          </label>
          <button
            disabled={!canGiveClue || clue.trim().length === 0 || count < 0}
            onClick={() => {
              onGiveClue(clue.trim(), count)
              setClue('')
            }}
          >
            Submit GiveClue
          </button>
        </div>
      ) : (
        <div className="control-group">
          <p>Click a highlighted tile to send Guess(index).</p>
          <button disabled={!canEndTurn} onClick={onEndTurn}>
            EndTurn
          </button>
        </div>
      )}

      <button className="danger" disabled={!canResign} onClick={onResign}>
        Resign
      </button>
    </section>
  )
}
