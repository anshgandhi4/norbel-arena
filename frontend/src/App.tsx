import { useEffect, useMemo, useState } from 'react'
import Board from './components/Board'
import ControlsPanel from './components/ControlsPanel'
import GameInfo from './components/GameInfo'
import MatchSetup from './components/MatchSetup'
import MoveHistory from './components/MoveHistory'
import { createMatch, fetchEvents, fetchObservation, submitMove } from './lib/api'
import type { MatchEvent, MatchView, Move } from './lib/types'

export default function App() {
  const [view, setView] = useState<MatchView | null>(null)
  const [events, setEvents] = useState<MatchEvent[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const matchId = view?.match_id
  const playerId = view?.player_id

  useEffect(() => {
    if (!matchId || !playerId) {
      return
    }

    let cancelled = false

    const poll = async () => {
      if (!view || view.meta.terminal || view.meta.is_human_turn) {
        return
      }
      try {
        const latest = await fetchObservation(matchId, playerId)
        if (!cancelled) {
          setView(latest)
          const ev = await fetchEvents(matchId)
          if (!cancelled) {
            setEvents(ev)
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
        }
      }
    }

    const timer = window.setInterval(poll, 900)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [matchId, playerId, view])

  const onCreateMatch = async (params: {
    humanPlayerId: string
    seed: number
    boardSize: number
    startingTeam: 'RED' | 'BLUE' | 'RANDOM'
  }) => {
    setLoading(true)
    setError(null)
    try {
      const players: Record<string, string> = {
        RED_OPERATIVE: 'random',
        RED_SPYMASTER: 'random',
        BLUE_OPERATIVE: 'random',
        BLUE_SPYMASTER: 'random'
      }
      players[params.humanPlayerId] = 'human'

      const config: Record<string, unknown> = {
        board_size: params.boardSize
      }
      if (params.startingTeam !== 'RANDOM') {
        config.starting_team = params.startingTeam
      }

      const created = await createMatch({
        seed: params.seed,
        config,
        players,
        human_player_id: params.humanPlayerId
      })
      setView(created)
      const ev = await fetchEvents(created.match_id)
      setEvents(ev)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const submit = async (move: Move) => {
    if (!view) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      const updated = await submitMove(view.match_id, view.player_id, move)
      setView(updated)
      const ev = await fetchEvents(updated.match_id)
      setEvents(ev)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      try {
        const refreshed = await fetchObservation(view.match_id, view.player_id)
        setView(refreshed)
      } catch {
        // no-op: original error is enough
      }
    } finally {
      setLoading(false)
    }
  }

  const canRenderGame = useMemo(() => !!view, [view])

  if (!canRenderGame || !view) {
    return (
      <main className="app setup-page">
        <MatchSetup onCreate={onCreateMatch} loading={loading} error={error} />
      </main>
    )
  }

  return (
    <main className="app game-page">
      <header className="topbar">
        <h2>Match {view.match_id}</h2>
        <button
          onClick={() => {
            setView(null)
            setEvents([])
            setError(null)
          }}
        >
          New Match
        </button>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="layout">
        <Board
          board={view.observation.board}
          legalMovesSpec={view.legal_moves_spec}
          isHumanTurn={view.meta.is_human_turn && !view.meta.terminal && !loading}
          onGuess={(index) => submit({ type: 'Guess', index })}
        />

        <aside className="sidebar">
          <GameInfo observation={view.observation} meta={view.meta} result={view.result} />
          <ControlsPanel
            observation={view.observation}
            legalMovesSpec={view.legal_moves_spec}
            isHumanTurn={view.meta.is_human_turn && !view.meta.terminal && !loading}
            onGiveClue={(clue, count) => submit({ type: 'GiveClue', clue, count })}
            onEndTurn={() => submit({ type: 'EndTurn' })}
            onResign={() => submit({ type: 'Resign' })}
          />
          <MoveHistory events={events} />
        </aside>
      </section>
    </main>
  )
}
