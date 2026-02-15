export type GameName = 'codenames' | 'wavelength'

export type Team = 'RED' | 'BLUE'
export type Role = 'OPERATIVE' | 'SPYMASTER' | 'PSYCHIC' | 'GUESSER'
export type Phase =
  | 'SPYMASTER_CLUE'
  | 'OPERATIVE_GUESSING'
  | 'CHOOSE_NUMBER'
  | 'CATEGORY_PROMPT'
  | 'PSYCHIC_RESPONSE'
  | 'NUMERIC_GUESS'
  | 'FINAL_ESTIMATE'

export type EvaluationMode = 'operator_eval' | 'spymaster_eval' | 'guesser_eval'

export type CodenamesPlayerId = 'RED_OPERATIVE' | 'RED_SPYMASTER' | 'BLUE_OPERATIVE' | 'BLUE_SPYMASTER'
export type WavelengthPlayerId = 'PSYCHIC' | 'GUESSER_ONE' | 'GUESSER_TWO'
export type PlayerId = CodenamesPlayerId | WavelengthPlayerId | string

export type AgentType =
  | 'human'
  | 'random'
  | 'openai'
  | 'anthropic'
  | 'perplexity'
  | 'local'
  | 'nemotron'

export interface PlayerConfig {
  type: AgentType
  model?: string
  backend?: string
  base_url?: string
  system_prompt?: string
}

export type Move =
  | { type: 'GiveClue'; clue: string; count: number }
  | { type: 'Guess'; index: number }
  | { type: 'EndTurn' }
  | { type: 'ChooseNumber'; value: number }
  | { type: 'AskCategory'; category: string }
  | { type: 'GiveAnswer'; answer: string }
  | { type: 'SubmitGuess'; value: number }
  | { type: 'SubmitFinalEstimate'; value: number }

export interface BoardTile {
  index: number
  word: string
  revealed: boolean
  revealed_color: Team | 'NEUTRAL' | 'ASSASSIN' | null
  assignment: Team | 'NEUTRAL' | 'ASSASSIN' | null
}

export interface CodenamesObservation {
  game: 'codenames'
  player_id: string
  role: 'OPERATIVE' | 'SPYMASTER'
  team: Team
  turn_team: Team
  phase: 'SPYMASTER_CLUE' | 'OPERATIVE_GUESSING'
  board: BoardTile[]
  current_clue: { clue: string; count: number } | null
  guesses_remaining: number
  public_counts: {
    RED_LEFT: number
    BLUE_LEFT: number
  }
  last_move_summary: string | null
  turn_index: number
  last_move?: Record<string, unknown> | null
}

export interface WavelengthExchange {
  round: number
  guesser_id: string
  category: string
  answer?: string | null
  guess?: number
}

export interface WavelengthObservation {
  game: 'wavelength'
  player_id: string
  role: 'PSYCHIC' | 'GUESSER'
  phase: 'CHOOSE_NUMBER' | 'CATEGORY_PROMPT' | 'PSYCHIC_RESPONSE' | 'NUMERIC_GUESS' | 'FINAL_ESTIMATE'
  current_player: string
  current_round: number
  max_rounds: number
  target_number: number | null
  history: WavelengthExchange[]
  pending_exchange: {
    round: number
    guesser_id: string
    category: string
    answer: string | null
  } | null
  guess_counts: Record<string, number>
  final_estimates: Record<string, number | null>
  winner: string | null
  termination_reason: string | null
  last_move_summary: string | null
  turn_index: number
  last_move?: Record<string, unknown> | null
}

export type Observation = CodenamesObservation | WavelengthObservation

export interface LegalMovesSpec {
  enumerated?: Move[]
  type?: string
  allowed?: {
    Guess?: { indices: number[] }
    EndTurn?: boolean
    GiveClue?: boolean | { count_min?: number; count_max?: number }
    ChooseNumber?: boolean | { min?: number; max?: number }
    AskCategory?: boolean
    GiveAnswer?: boolean
    SubmitGuess?: boolean | { min?: number; max?: number }
    SubmitFinalEstimate?: boolean | { min?: number; max?: number }
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface MatchMeta {
  game: GameName
  match_id: string
  seed: number
  evaluation_mode?: EvaluationMode
  current_player: string | null
  is_human_turn: boolean
  terminal: boolean
  human_players: string[]
  replay_turn: number
  max_turn: number
  is_live: boolean
  player_configs: Record<string, PlayerConfig>
}

export interface MatchResult {
  winner: string | null
  termination_reason: string
  turns: number
  scores: Record<string, number>
  details?: string | null
}

export interface MatchView {
  match_id: string
  player_id: string
  observation: Observation
  legal_moves_spec: LegalMovesSpec
  meta: MatchMeta
  last_event?: MatchEvent
  result?: MatchResult
}

export interface MatchEvent {
  event_type: string
  game_id: string
  turn: number
  timestamp_ms: number
  payload: Record<string, unknown>
}

export interface AgentReportCard {
  matches: number
  wins: number
  losses: number
  draws: number
  total_turns: number
  avg_turns: number
  win_rate: number
  elo?: number
  terminations: Record<string, number>
  seats: Record<string, number>
  last_played: string | null
}

export interface ReportCards {
  version: number
  updated_at: string
  games: Record<string, Record<string, AgentReportCard>>
}
