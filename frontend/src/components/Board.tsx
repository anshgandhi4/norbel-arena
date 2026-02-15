import type { BoardTile, LegalMovesSpec } from '../lib/types'

interface BoardProps {
  board: BoardTile[]
  legalMovesSpec: LegalMovesSpec
  isHumanTurn: boolean
  onGuess: (index: number) => void
}

function tileClass(tile: BoardTile): string {
  if (tile.revealed) {
    return `tile revealed ${String(tile.revealed_color ?? '').toLowerCase()}`
  }
  if (tile.assignment) {
    return `tile hinted ${String(tile.assignment).toLowerCase()}`
  }
  return 'tile hidden'
}

export default function Board({ board, legalMovesSpec, isHumanTurn, onGuess }: BoardProps) {
  const guessMask = legalMovesSpec.allowed?.Guess
  const guessIndices = new Set(
    guessMask && typeof guessMask === 'object' && 'indices' in guessMask && Array.isArray(guessMask.indices)
      ? guessMask.indices
      : []
  )
  const columns = Math.max(4, Math.round(Math.sqrt(board.length)))

  return (
    <section className="board" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {board.map((tile) => {
        const canGuess = isHumanTurn && guessIndices.has(tile.index)
        return (
          <button
            key={tile.index}
            className={`${tileClass(tile)} ${canGuess ? 'clickable' : ''}`}
            disabled={!canGuess}
            onClick={() => onGuess(tile.index)}
          >
            <span className="word">{tile.word}</span>
          </button>
        )
      })}
    </section>
  )
}
