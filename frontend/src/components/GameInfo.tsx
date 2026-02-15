import type { MatchMeta, MatchResult, Observation } from '../lib/types'

interface GameInfoProps {
  observation: Observation
  meta: MatchMeta
  result?: MatchResult
}

export default function GameInfo({ observation, meta, result }: GameInfoProps) {
  return (
    <section className="panel card">
      <h3>Game Info</h3>
      <dl>
        <dt>Player</dt>
        <dd>{observation.player_id}</dd>
        <dt>Team / Role</dt>
        <dd>
          {observation.team} / {observation.role}
        </dd>
        <dt>Turn</dt>
        <dd>{observation.turn_team}</dd>
        <dt>Phase</dt>
        <dd>{observation.phase}</dd>
        <dt>Current clue</dt>
        <dd>{observation.current_clue ? `${observation.current_clue.clue} (${observation.current_clue.count})` : '-'}</dd>
        <dt>Guesses remaining</dt>
        <dd>{observation.guesses_remaining}</dd>
        <dt>RED left</dt>
        <dd>{observation.public_counts.RED_LEFT}</dd>
        <dt>BLUE left</dt>
        <dd>{observation.public_counts.BLUE_LEFT}</dd>
        <dt>Human turn</dt>
        <dd>{meta.is_human_turn ? 'Yes' : 'No'}</dd>
      </dl>

      {result ? (
        <div className="result-box">
          <strong>Game Over</strong>
          <p>Winner: {result.winner ?? 'Draw'}</p>
          <p>Reason: {result.termination_reason}</p>
        </div>
      ) : null}
    </section>
  )
}
