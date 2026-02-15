import { useMemo, useState } from 'react'
import type { LegalMovesSpec, Move, WavelengthObservation } from '../lib/types'

interface WavelengthControlsProps {
  observation: WavelengthObservation
  legalMovesSpec: LegalMovesSpec
  isHumanTurn: boolean
  onSubmitMove: (move: Move) => void
}

function isActionEnabled(legalMovesSpec: LegalMovesSpec, action: string): boolean {
  const allowed = legalMovesSpec.allowed?.[action]
  if (typeof allowed === 'boolean') {
    return allowed
  }
  return allowed !== undefined
}

function numericBounds(legalMovesSpec: LegalMovesSpec, action: string): { min: number; max: number } {
  const allowed = legalMovesSpec.allowed?.[action]
  if (allowed && typeof allowed === 'object') {
    const numericAllowed = allowed as { min?: unknown; max?: unknown }
    const min = typeof numericAllowed.min === 'number' ? numericAllowed.min : 1
    const max = typeof numericAllowed.max === 'number' ? numericAllowed.max : 100
    return { min, max }
  }
  return { min: 1, max: 100 }
}

export default function WavelengthControls({
  observation,
  legalMovesSpec,
  isHumanTurn,
  onSubmitMove
}: WavelengthControlsProps) {
  const [category, setCategory] = useState('car brand')
  const [answer, setAnswer] = useState('')
  const [value, setValue] = useState(50)

  const canChooseNumber = useMemo(
    () => isHumanTurn && isActionEnabled(legalMovesSpec, 'ChooseNumber'),
    [isHumanTurn, legalMovesSpec]
  )
  const canAskCategory = useMemo(
    () => isHumanTurn && isActionEnabled(legalMovesSpec, 'AskCategory'),
    [isHumanTurn, legalMovesSpec]
  )
  const canGiveAnswer = useMemo(
    () => isHumanTurn && isActionEnabled(legalMovesSpec, 'GiveAnswer'),
    [isHumanTurn, legalMovesSpec]
  )
  const canSubmitGuess = useMemo(
    () => isHumanTurn && isActionEnabled(legalMovesSpec, 'SubmitGuess'),
    [isHumanTurn, legalMovesSpec]
  )
  const canSubmitFinal = useMemo(
    () => isHumanTurn && isActionEnabled(legalMovesSpec, 'SubmitFinalEstimate'),
    [isHumanTurn, legalMovesSpec]
  )

  const chooseBounds = numericBounds(legalMovesSpec, 'ChooseNumber')
  const guessBounds = numericBounds(legalMovesSpec, 'SubmitGuess')
  const finalBounds = numericBounds(legalMovesSpec, 'SubmitFinalEstimate')
  const submitPsychicAnswer = () => {
    const trimmed = answer.trim()
    if (!trimmed || !canGiveAnswer) {
      return
    }
    onSubmitMove({ type: 'GiveAnswer', answer: trimmed })
    setAnswer('')
  }

  return (
    <section className="panel card">
      <h3>Controls</h3>

      {canChooseNumber ? (
        <div className="control-group">
          <label>
            Hidden number
            <input
              type="number"
              min={chooseBounds.min}
              max={chooseBounds.max}
              value={value}
              onChange={(event) => setValue(Number(event.target.value))}
            />
          </label>
          <button onClick={() => onSubmitMove({ type: 'ChooseNumber', value })}>Choose Number</button>
        </div>
      ) : null}

      {canAskCategory ? (
        <div className="control-group">
          <label>
            Category prompt
            <input value={category} onChange={(event) => setCategory(event.target.value)} />
          </label>
          <button disabled={category.trim().length === 0} onClick={() => onSubmitMove({ type: 'AskCategory', category: category.trim() })}>
            Ask Category
          </button>
        </div>
      ) : null}

      {canGiveAnswer ? (
        <div className="control-group">
          <label>
            Psychic answer
            <input
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              onKeyDown={(event) => {
                if (event.key !== 'Enter') {
                  return
                }
                event.preventDefault()
                submitPsychicAnswer()
              }}
            />
          </label>
          <button disabled={answer.trim().length === 0} onClick={submitPsychicAnswer}>
            Give Answer
          </button>
        </div>
      ) : null}

      {canSubmitGuess ? (
        <div className="control-group">
          <label>
            Numeric guess
            <input
              type="number"
              min={guessBounds.min}
              max={guessBounds.max}
              value={value}
              onChange={(event) => setValue(Number(event.target.value))}
            />
          </label>
          <button onClick={() => onSubmitMove({ type: 'SubmitGuess', value })}>Submit Guess</button>
        </div>
      ) : null}

      {canSubmitFinal ? (
        <div className="control-group">
          <label>
            Final estimate
            <input
              type="number"
              min={finalBounds.min}
              max={finalBounds.max}
              value={value}
              onChange={(event) => setValue(Number(event.target.value))}
            />
          </label>
          <button onClick={() => onSubmitMove({ type: 'SubmitFinalEstimate', value })}>Submit Final Estimate</button>
        </div>
      ) : null}

      {!canChooseNumber && !canAskCategory && !canGiveAnswer && !canSubmitGuess && !canSubmitFinal ? (
        <p className="control-help">Waiting for the active player to move.</p>
      ) : null}

      <p className="muted">Current phase: {observation.phase}</p>
    </section>
  )
}
