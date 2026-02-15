import { Fragment, useEffect, useMemo, useState } from 'react'
import { formatPlayerId } from '../lib/format'
import type { AgentType, EvaluationMode, GameName, PlayerConfig } from '../lib/types'

export interface SetupSelections {
  game: GameName
  startingTeam?: 'RED' | 'BLUE' | 'RANDOM'
  evaluationMode: EvaluationMode
  players: Record<string, PlayerConfig>
  viewerPlayerId: string
}

interface MatchSetupProps {
  onCreate: (params: SetupSelections) => Promise<void>
  loading: boolean
  error: string | null
  initialSelections?: SetupSelections | null
}

const CODENAMES_PLAYER_IDS = ['RED_SPYMASTER', 'RED_OPERATIVE', 'BLUE_SPYMASTER', 'BLUE_OPERATIVE'] as const
const WAVELENGTH_PLAYER_IDS = ['PSYCHIC', 'GUESSER_ONE', 'GUESSER_TWO'] as const
const GAME_PLAYER_IDS: Record<GameName, readonly string[]> = {
  codenames: CODENAMES_PLAYER_IDS,
  wavelength: WAVELENGTH_PLAYER_IDS
}

const AGENT_TYPES: AgentType[] = ['human', 'openai', 'anthropic', 'perplexity', 'local', 'nemotron']

const DEFAULT_PLAYERS_BY_GAME: Record<GameName, Record<string, PlayerConfig>> = {
  codenames: {
    RED_SPYMASTER: { type: 'human' },
    RED_OPERATIVE: { type: 'human' },
    BLUE_SPYMASTER: { type: 'human' },
    BLUE_OPERATIVE: { type: 'human' }
  },
  wavelength: {
    PSYCHIC: { type: 'human' },
    GUESSER_ONE: { type: 'openai', model: 'gpt-4o-mini' },
    GUESSER_TWO: { type: 'anthropic', model: 'claude-sonnet-4-20250514' }
  }
}

const DEFAULT_VIEWER_BY_GAME: Record<GameName, string> = {
  codenames: 'RED_SPYMASTER',
  wavelength: 'PSYCHIC'
}

const DEFAULT_EVAL_BY_GAME: Record<GameName, EvaluationMode> = {
  codenames: 'operator_eval',
  wavelength: 'guesser_eval'
}

const MODEL_OPTIONS: Partial<Record<AgentType, string[]>> = {
  openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-5-mini', 'gpt-5'],
  anthropic: ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-3-7-sonnet-latest'],
  perplexity: ['sonar', 'sonar-pro', 'sonar-reasoning'],
  local: ['meta-llama/Llama-3.1-8B-Instruct', 'mistralai/Ministral-3-8B-Instruct-2512-BF16'],
  nemotron: [
    'nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1',
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1',
    'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8'
  ]
}

const NEEDS_MODEL = new Set<AgentType>(['openai', 'anthropic', 'perplexity', 'local', 'nemotron'])
const LOCAL_LIKE = new Set<AgentType>(['local', 'nemotron'])

function clonePlayers(players: Record<string, PlayerConfig>): Record<string, PlayerConfig> {
  return Object.fromEntries(Object.entries(players).map(([key, value]) => [key, { ...value }]))
}

function defaultBackendForType(type: AgentType): string | undefined {
  if (LOCAL_LIKE.has(type)) {
    return 'transformers'
  }
  return undefined
}

function defaultModelForType(type: AgentType): string {
  const options = MODEL_OPTIONS[type]
  if (options && options.length > 0) {
    return options[0]
  }
  return ''
}

function normalizeModelForType(type: AgentType, existingModel?: string): string | undefined {
  if (!NEEDS_MODEL.has(type)) {
    return undefined
  }
  const options = MODEL_OPTIONS[type] ?? []
  if (existingModel && options.includes(existingModel)) {
    return existingModel
  }
  return defaultModelForType(type)
}

function seatLabelClass(playerId: string): string {
  if (playerId.startsWith('RED')) {
    return 'seat-label red'
  }
  if (playerId.startsWith('BLUE')) {
    return 'seat-label blue'
  }
  return 'seat-label'
}

function llmSignature(config: PlayerConfig): string | null {
  if (!NEEDS_MODEL.has(config.type)) {
    return null
  }

  const model = (config.model ?? '').trim().toLowerCase()
  if (!model) {
    return null
  }

  if (LOCAL_LIKE.has(config.type)) {
    const backendDefault = defaultBackendForType(config.type) ?? 'ollama'
    const backend = (config.backend ?? backendDefault).trim().toLowerCase() || backendDefault
    return `${config.type}|${backend}|${model}`
  }

  return `${config.type}|${model}`
}

function applyCodenamesMirroring(
  players: Record<string, PlayerConfig>,
  evaluationMode: EvaluationMode
): Record<string, PlayerConfig> {
  const next = clonePlayers(players)
  if (evaluationMode === 'operator_eval') {
    next.BLUE_SPYMASTER = { ...next.RED_SPYMASTER }
  } else if (evaluationMode === 'spymaster_eval') {
    next.BLUE_OPERATIVE = { ...next.RED_OPERATIVE }
  }
  return next
}

function initialGameFromSelections(initialSelections?: SetupSelections | null): GameName {
  return initialSelections?.game ?? 'codenames'
}

function initialPlayersByGame(initialSelections?: SetupSelections | null): Record<GameName, Record<string, PlayerConfig>> {
  const baseline = {
    codenames: clonePlayers(DEFAULT_PLAYERS_BY_GAME.codenames),
    wavelength: clonePlayers(DEFAULT_PLAYERS_BY_GAME.wavelength)
  }
  if (!initialSelections) {
    return baseline
  }
  baseline[initialSelections.game] = clonePlayers(initialSelections.players)
  return baseline
}

function initialViewerByGame(initialSelections?: SetupSelections | null): Record<GameName, string> {
  const baseline = {
    codenames: DEFAULT_VIEWER_BY_GAME.codenames,
    wavelength: DEFAULT_VIEWER_BY_GAME.wavelength
  }
  if (!initialSelections) {
    return baseline
  }
  baseline[initialSelections.game] = initialSelections.viewerPlayerId
  return baseline
}

function initialEvaluationByGame(initialSelections?: SetupSelections | null): Record<GameName, EvaluationMode> {
  const baseline = {
    codenames: DEFAULT_EVAL_BY_GAME.codenames,
    wavelength: DEFAULT_EVAL_BY_GAME.wavelength
  }
  if (!initialSelections) {
    return baseline
  }
  baseline[initialSelections.game] = initialSelections.evaluationMode
  return baseline
}

export default function MatchSetup({ onCreate, loading, error, initialSelections }: MatchSetupProps) {
  const [game, setGame] = useState<GameName>(() => initialGameFromSelections(initialSelections))
  const [playersByGame, setPlayersByGame] = useState<Record<GameName, Record<string, PlayerConfig>>>(() =>
    initialPlayersByGame(initialSelections)
  )
  const [viewerByGame, setViewerByGame] = useState<Record<GameName, string>>(() =>
    initialViewerByGame(initialSelections)
  )
  const [evaluationModeByGame, setEvaluationModeByGame] = useState<Record<GameName, EvaluationMode>>(() =>
    initialEvaluationByGame(initialSelections)
  )
  const [startingTeam, setStartingTeam] = useState<'RED' | 'BLUE' | 'RANDOM'>(
    () => initialSelections?.startingTeam ?? 'RANDOM'
  )
  const [toastMessage, setToastMessage] = useState<string | null>(null)
  const lockDropdowns = loading && game === 'codenames'

  const playerIds = GAME_PLAYER_IDS[game]
  const rawPlayers = playersByGame[game]
  const evaluationMode = evaluationModeByGame[game]

  const effectivePlayers = useMemo(() => {
    if (game === 'codenames') {
      return applyCodenamesMirroring(rawPlayers, evaluationMode)
    }
    return clonePlayers(rawPlayers)
  }, [game, rawPlayers, evaluationMode])

  const lockedSeat = useMemo(() => {
    if (game !== 'codenames') {
      return null
    }
    return evaluationMode === 'operator_eval' ? 'BLUE_SPYMASTER' : 'BLUE_OPERATIVE'
  }, [game, evaluationMode])

  const duplicateModelMessage = useMemo(() => {
    if (game === 'codenames') {
      if (evaluationMode === 'operator_eval') {
        const redSig = llmSignature(effectivePlayers.RED_OPERATIVE)
        const blueSig = llmSignature(effectivePlayers.BLUE_OPERATIVE)
        if (redSig !== null && blueSig !== null && redSig === blueSig) {
          return 'Operators must be different in operator_eval mode'
        }
        return null
      }

      const redSig = llmSignature(effectivePlayers.RED_SPYMASTER)
      const blueSig = llmSignature(effectivePlayers.BLUE_SPYMASTER)
      if (redSig !== null && blueSig !== null && redSig === blueSig) {
        return 'Spymasters must be different in spymaster_eval mode'
      }
      return null
    }

    const guesserOneSig = llmSignature(effectivePlayers.GUESSER_ONE)
    const guesserTwoSig = llmSignature(effectivePlayers.GUESSER_TWO)
    if (guesserOneSig !== null && guesserTwoSig !== null && guesserOneSig === guesserTwoSig) {
      return 'Guessers must be different in guesser_eval mode'
    }
    return null
  }, [game, effectivePlayers, evaluationMode])

  useEffect(() => {
    if (!duplicateModelMessage) {
      return
    }
    setToastMessage(duplicateModelMessage)
  }, [duplicateModelMessage])

  useEffect(() => {
    if (!error) {
      return
    }
    setToastMessage(error)
  }, [error])

  useEffect(() => {
    if (!toastMessage) {
      return
    }
    const timeout = window.setTimeout(() => setToastMessage(null), 3200)
    return () => window.clearTimeout(timeout)
  }, [toastMessage])

  const setPlayerType = (playerId: string, type: AgentType) => {
    if (lockedSeat === playerId) {
      return
    }

    setPlayersByGame((prev) => {
      const nextByGame = { ...prev }
      const nextPlayers = clonePlayers(nextByGame[game])
      const current = nextPlayers[playerId]
      const isLocalLike = type === 'local' || type === 'nemotron'
      const defaultBackend = defaultBackendForType(type)
      nextPlayers[playerId] = {
        ...current,
        type,
        model: normalizeModelForType(type, current?.model),
        backend: isLocalLike ? current?.backend ?? defaultBackend : undefined,
        base_url: type === 'local' ? current?.base_url : undefined
      }
      nextByGame[game] = nextPlayers
      return nextByGame
    })
  }

  const setPlayerField = (playerId: string, field: keyof PlayerConfig, value: string) => {
    if (lockedSeat === playerId) {
      return
    }

    setPlayersByGame((prev) => {
      const nextByGame = { ...prev }
      const nextPlayers = clonePlayers(nextByGame[game])
      nextPlayers[playerId] = {
        ...nextPlayers[playerId],
        [field]: value
      }
      nextByGame[game] = nextPlayers
      return nextByGame
    })
  }

  const setViewerPlayerId = (playerId: string) => {
    setViewerByGame((prev) => ({
      ...prev,
      [game]: playerId
    }))
  }

  const setEvaluationMode = (mode: EvaluationMode) => {
    setEvaluationModeByGame((prev) => ({
      ...prev,
      [game]: mode
    }))
  }

  return (
    <section className="setup-card wide">
      <header className="setup-hero">
        <h1>Match Setup</h1>
        <p>Pick the game, assign agents to seats, choose a viewer perspective, and start the arena run.</p>
      </header>

      <div className="setup-inline">
        <label>
          Game
          <select value={game} onChange={(event) => setGame(event.target.value as GameName)} disabled={lockDropdowns}>
            <option value="codenames">codenames</option>
            <option value="wavelength">wavelength</option>
          </select>
        </label>

        <label>
          Evaluation mode
          {game === 'codenames' ? (
            <select
              value={evaluationMode}
              onChange={(event) => setEvaluationMode(event.target.value as EvaluationMode)}
              disabled={lockDropdowns}
            >
              <option value="operator_eval">operator_eval</option>
              <option value="spymaster_eval">spymaster_eval</option>
            </select>
          ) : (
            <input value="guesser_eval" disabled />
          )}
        </label>

        <label>
          Viewer perspective
          <select value={viewerByGame[game]} onChange={(event) => setViewerPlayerId(event.target.value)} disabled={lockDropdowns}>
            {playerIds.map((id) => (
              <option key={id} value={id}>
                {formatPlayerId(id)}
              </option>
            ))}
          </select>
        </label>

        {game === 'codenames' ? (
          <label>
            Starting team
            <select
              value={startingTeam}
              onChange={(event) => setStartingTeam(event.target.value as 'RED' | 'BLUE' | 'RANDOM')}
              disabled={lockDropdowns}
            >
              <option value="RANDOM">Random</option>
              <option value="RED">RED</option>
              <option value="BLUE">BLUE</option>
            </select>
          </label>
        ) : null}
      </div>

      <div className="seat-grid">
        <div className="seat-head">Seat</div>
        <div className="seat-head">Controller</div>
        <div className="seat-head">Model</div>

        {playerIds.map((playerId) => {
          const config = effectivePlayers[playerId]
          const needsModel = NEEDS_MODEL.has(config.type)
          const modelOptions = MODEL_OPTIONS[config.type] ?? []
          const isLocked = lockedSeat === playerId

          return (
            <Fragment key={playerId}>
              <div key={`${playerId}-label`} className={`seat-cell${isLocked ? ' mirrored-seat' : ''}`}>
                <span className={seatLabelClass(playerId)}>{formatPlayerId(playerId)}</span>
              </div>
              <div key={`${playerId}-type`} className={`seat-cell${isLocked ? ' mirrored-seat' : ''}`}>
                <select
                  className={isLocked ? 'mirrored-control' : undefined}
                  value={config.type}
                  onChange={(event) => setPlayerType(playerId, event.target.value as AgentType)}
                  disabled={isLocked || lockDropdowns}
                >
                  {AGENT_TYPES.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </div>
              <div key={`${playerId}-config`} className={`seat-cell${isLocked ? ' mirrored-seat' : ''}`}>
                {needsModel ? (
                  <div className="inline-fields">
                    <select
                      className={isLocked ? 'mirrored-control' : undefined}
                      value={config.model ?? defaultModelForType(config.type)}
                      onChange={(event) => setPlayerField(playerId, 'model', event.target.value)}
                      disabled={isLocked || lockDropdowns}
                    >
                      {modelOptions.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <span className="muted">-</span>
                )}
              </div>
            </Fragment>
          )
        })}
      </div>

      <button
        className="primary-button"
        disabled={loading || duplicateModelMessage !== null}
        onClick={() => {
          onCreate({
            game,
            startingTeam: game === 'codenames' ? startingTeam : undefined,
            evaluationMode,
            players: effectivePlayers,
            viewerPlayerId: viewerByGame[game]
          })
        }}
      >
        {loading ? 'Creating...' : 'Create Match'}
      </button>

      {toastMessage ? (
        <div className="toast-banner" role="status" aria-live="polite">
          {toastMessage}
        </div>
      ) : null}
    </section>
  )
}
