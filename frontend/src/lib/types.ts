export type Team = 'RED' | 'BLUE'
export type Role = 'OPERATIVE' | 'SPYMASTER'
export type Phase = 'SPYMASTER_CLUE' | 'OPERATIVE_GUESSING'

export type Move =
  | { type: 'GiveClue'; clue: string; count: number }
  | { type: 'Guess'; index: number }
  | { type: 'EndTurn' }
  | { type: 'Resign' }

export interface BoardTile {
  index: number
  word: string
  revealed: boolean
  revealed_color: Team | 'NEUTRAL' | 'ASSASSIN' | null
  assignment: Team | 'NEUTRAL' | 'ASSASSIN' | null
}

export interface Observation {
  game: string
  player_id: string
  role: Role
  team: Team
  turn_team: Team
  phase: Phase
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

export interface LegalMovesSpec {
  enumerated?: Move[]
  type?: string
  allowed?: {
    Guess?: { indices: number[] }
    EndTurn?: boolean
    GiveClue?: boolean | { count_min?: number; count_max?: number }
    Resign?: boolean
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface MatchMeta {
  match_id: string
  seed: number
  current_player: string | null
  is_human_turn: boolean
  terminal: boolean
  human_players: string[]
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
