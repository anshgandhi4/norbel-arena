import { useEffect, useMemo, useState } from 'react'
import Board from './components/Board'
import ControlsPanel from './components/ControlsPanel'
import GameInfo from './components/GameInfo'
import MatchSetup from './components/MatchSetup'
import type { SetupSelections } from './components/MatchSetup'
import MoveHistory from './components/MoveHistory'
import ReplayControls from './components/ReplayControls'
import AppHeader, { type Theme } from './components/AppHeader'
import WavelengthControls from './components/WavelengthControls'
import WavelengthTranscript from './components/WavelengthTranscript'
import { createMatch, fetchEvents, fetchObservation, submitMove } from './lib/api'
import type {
  CodenamesObservation,
  EvaluationMode,
  GameName,
  MatchEvent,
  MatchView,
  Move,
  PlayerConfig,
  WavelengthObservation
} from './lib/types'

const THEME_STORAGE_KEY = 'treehacks-ui-theme'

function resolveInitialTheme(): Theme {
  if (typeof window === 'undefined') {
    return 'dark'
  }
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
  return storedTheme === 'light' ? 'light' : 'dark'
}

export default function App() {
  const [view, setView] = useState<MatchView | null>(null)
  const [events, setEvents] = useState<MatchEvent[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(() => resolveInitialTheme())
  const [lastSetupSelections, setLastSetupSelections] = useState<SetupSelections | null>(null)

  const matchId = view?.match_id
  const playerId = view?.player_id

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    document.documentElement.style.colorScheme = theme
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  useEffect(() => {
    if (!matchId || !playerId || !view?.meta.is_live) {
      return
    }

    let cancelled = false

    const poll = async () => {
      if (!view || view.meta.terminal || view.meta.is_human_turn || !view.meta.is_live) {
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
    game: GameName
    startingTeam?: 'RED' | 'BLUE' | 'RANDOM'
    evaluationMode: EvaluationMode
    players: Record<string, PlayerConfig>
    viewerPlayerId: string
  }) => {
    setLoading(true)
    setError(null)
    setLastSetupSelections({
      game: params.game,
      startingTeam: params.startingTeam,
      evaluationMode: params.evaluationMode,
      players: Object.fromEntries(
        Object.entries(params.players).map(([playerId, config]) => [playerId, { ...config }])
      ),
      viewerPlayerId: params.viewerPlayerId
    })
    try {
      const config: Record<string, unknown> = {
        evaluation_mode: params.evaluationMode
      }
      if (params.game === 'codenames' && params.startingTeam && params.startingTeam !== 'RANDOM') {
        config.starting_team = params.startingTeam
      }

      const humanPlayers = Object.entries(params.players)
        .filter(([, playerConfig]) => playerConfig.type === 'human')
        .map(([playerIdValue]) => playerIdValue)

      const created = await createMatch({
        game: params.game,
        config,
        players: params.players,
        human_player_id: humanPlayers.length === 1 ? humanPlayers[0] : undefined,
        viewer_player_id: params.viewerPlayerId
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

  const refreshView = async (targetTurn?: number) => {
    if (!view) {
      return
    }
    const refreshed = await fetchObservation(view.match_id, view.player_id, targetTurn)
    setView(refreshed)
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
        await refreshView()
      } catch {
        // no-op: original error is enough
      }
    } finally {
      setLoading(false)
    }
  }

  const goToTurn = async (turn: number) => {
    if (!view) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await refreshView(turn)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const goLive = async () => {
    if (!view) {
      return
    }
    setLoading(true)
    setError(null)
    try {
      await refreshView()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const canRenderGame = useMemo(() => !!view, [view])
  const canAct = !!view && view.meta.is_human_turn && view.meta.is_live && !view.meta.terminal && !loading
  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))

  if (!canRenderGame || !view) {
    return (
      <main className="app setup-page">
        <AppHeader theme={theme} onToggleTheme={toggleTheme} leftLinks={[{ href: '/leaderboard', label: 'Leaderboards' }]} />
        <div className="setup-stack">
          <MatchSetup
            onCreate={onCreateMatch}
            loading={loading}
            error={error}
            initialSelections={lastSetupSelections}
          />
        </div>
      </main>
    )
  }

  const isCodenames = view.observation.game === 'codenames'

  return (
    <main className="app game-page">
      <AppHeader theme={theme} onToggleTheme={toggleTheme} leftLinks={[{ href: '/leaderboard', label: 'Leaderboards' }]} />

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="layout">
        <div className="board-column">
          {isCodenames ? (
            <Board
              board={(view.observation as CodenamesObservation).board}
              legalMovesSpec={view.legal_moves_spec}
              isHumanTurn={canAct}
              onGuess={(index) => submit({ type: 'Guess', index })}
            />
          ) : (
            <WavelengthTranscript observation={view.observation as WavelengthObservation} />
          )}

          <div className="board-actions">
            <button
              className="board-new-match-button"
              onClick={() => {
                setView(null)
                setEvents([])
                setError(null)
              }}
            >
              New Match
            </button>
          </div>
        </div>

        <aside className="sidebar">
          <ReplayControls
            replayTurn={view.meta.replay_turn}
            maxTurn={view.meta.max_turn}
            isLive={view.meta.is_live}
            onPrev={() => goToTurn(view.meta.replay_turn - 1)}
            onNext={() => goToTurn(view.meta.replay_turn + 1)}
            onLive={goLive}
          />
          <GameInfo observation={view.observation} meta={view.meta} result={view.result} />

          {isCodenames ? (
            <ControlsPanel
              observation={view.observation as CodenamesObservation}
              legalMovesSpec={view.legal_moves_spec}
              isHumanTurn={canAct}
              onGiveClue={(clue, count) => submit({ type: 'GiveClue', clue, count })}
              onEndTurn={() => submit({ type: 'EndTurn' })}
            />
          ) : (
            <WavelengthControls
              observation={view.observation as WavelengthObservation}
              legalMovesSpec={view.legal_moves_spec}
              isHumanTurn={canAct}
              onSubmitMove={submit}
            />
          )}

          <MoveHistory events={events} selectedTurn={view.meta.replay_turn} onSelectTurn={goToTurn} />
        </aside>
      </section>
    </main>
  )
}
