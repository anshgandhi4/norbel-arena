import { formatPlayerId } from '../lib/format'
import type {
  CodenamesObservation,
  MatchMeta,
  MatchResult,
  Observation,
  WavelengthObservation
} from '../lib/types'

interface GameInfoProps {
  observation: Observation
  meta: MatchMeta
  result?: MatchResult
}

function CodenamesInfo({ observation, meta }: { observation: CodenamesObservation; meta: MatchMeta }) {
  return (
    <>
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
      <dt>Mode</dt>
      <dd>{meta.is_live ? 'Live' : 'Replay'}</dd>
      <dt>Replay turn</dt>
      <dd>
        {meta.replay_turn} / {meta.max_turn}
      </dd>
    </>
  )
}

function WavelengthInfo({ observation, meta }: { observation: WavelengthObservation; meta: MatchMeta }) {
  return (
    <>
      <dt>Role</dt>
      <dd>{observation.role}</dd>
      <dt>Phase</dt>
      <dd>{observation.phase}</dd>
      <dt>Current player</dt>
      <dd>{formatPlayerId(observation.current_player)}</dd>
      <dt>Round</dt>
      <dd>
        {observation.current_round} / {observation.max_rounds}
      </dd>
      <dt>Target number</dt>
      <dd>{observation.target_number ?? 'Hidden'}</dd>
      <dt>Guesser One attempts</dt>
      <dd>{observation.guess_counts.GUESSER_ONE ?? 0}</dd>
      <dt>Guesser Two attempts</dt>
      <dd>{observation.guess_counts.GUESSER_TWO ?? 0}</dd>
      <dt>Human turn</dt>
      <dd>{meta.is_human_turn ? 'Yes' : 'No'}</dd>
      <dt>Mode</dt>
      <dd>{meta.is_live ? 'Live' : 'Replay'}</dd>
      <dt>Replay turn</dt>
      <dd>
        {meta.replay_turn} / {meta.max_turn}
      </dd>
    </>
  )
}

export default function GameInfo({ observation, meta, result }: GameInfoProps) {
  return (
    <section className="panel card">
      <h3>Game Info</h3>
      <dl>
        <dt>Game</dt>
        <dd>{observation.game}</dd>
        <dt>Match ID</dt>
        <dd>{meta.match_id}</dd>
        <dt>Player</dt>
        <dd>{formatPlayerId(observation.player_id)}</dd>

        {observation.game === 'codenames' ? (
          <CodenamesInfo observation={observation} meta={meta} />
        ) : (
          <WavelengthInfo observation={observation} meta={meta} />
        )}
      </dl>

      {result ? (
        <div className="result-box">
          <strong>Game Over</strong>
          <p>Winner: {result.winner ? formatPlayerId(result.winner) : 'Draw'}</p>
          <p>Reason: {result.termination_reason}</p>
        </div>
      ) : null}
    </section>
  )
}
