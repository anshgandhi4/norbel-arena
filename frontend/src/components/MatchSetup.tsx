import { useState } from 'react'

interface MatchSetupProps {
  onCreate: (params: {
    humanPlayerId: string
    seed: number
    boardSize: number
    startingTeam: 'RED' | 'BLUE' | 'RANDOM'
  }) => Promise<void>
  loading: boolean
  error: string | null
}

const PLAYER_IDS = ['RED_OPERATIVE', 'RED_SPYMASTER', 'BLUE_OPERATIVE', 'BLUE_SPYMASTER']

export default function MatchSetup({ onCreate, loading, error }: MatchSetupProps) {
  const [humanPlayerId, setHumanPlayerId] = useState<string>('RED_OPERATIVE')
  const [seed, setSeed] = useState<number>(123)
  const [boardSize, setBoardSize] = useState<number>(25)
  const [startingTeam, setStartingTeam] = useState<'RED' | 'BLUE' | 'RANDOM'>('RANDOM')

  return (
    <section className="setup-card">
      <h1>Codenames Arena</h1>
      <p>Play as one seat while the other seats are controlled by baseline AI.</p>

      <label>
        Human seat
        <select value={humanPlayerId} onChange={(e) => setHumanPlayerId(e.target.value)}>
          {PLAYER_IDS.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </label>

      <label>
        Seed
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value))}
        />
      </label>

      <label>
        Board size
        <input
          type="number"
          min={4}
          value={boardSize}
          onChange={(e) => setBoardSize(Number(e.target.value))}
        />
      </label>

      <label>
        Starting team
        <select value={startingTeam} onChange={(e) => setStartingTeam(e.target.value as 'RED' | 'BLUE' | 'RANDOM')}>
          <option value="RANDOM">Random</option>
          <option value="RED">RED</option>
          <option value="BLUE">BLUE</option>
        </select>
      </label>

      <button disabled={loading} onClick={() => onCreate({ humanPlayerId, seed, boardSize, startingTeam })}>
        {loading ? 'Creating...' : 'Create Match'}
      </button>

      {error ? <p className="error-text">{error}</p> : null}
    </section>
  )
}
