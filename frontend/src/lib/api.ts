import type { MatchEvent, MatchView, Move } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {})
    }
  })

  if (!response.ok) {
    const text = await response.text()
    try {
      const data = JSON.parse(text)
      if (typeof data?.detail === 'string') {
        throw new Error(data.detail)
      }
      if (data?.detail && typeof data.detail === 'object' && typeof data.detail.error === 'string') {
        throw new Error(data.detail.error)
      }
      throw new Error(text)
    } catch {
      throw new Error(text || `HTTP ${response.status}`)
    }
  }

  return response.json() as Promise<T>
}

export async function createMatch(params: {
  seed: number
  config: Record<string, unknown>
  players: Record<string, string>
  human_player_id: string
}): Promise<MatchView> {
  return request<MatchView>('/api/match/new', {
    method: 'POST',
    body: JSON.stringify(params)
  })
}

export async function fetchObservation(matchId: string, playerId: string): Promise<MatchView> {
  const query = new URLSearchParams({ player_id: playerId })
  return request<MatchView>(`/api/match/${matchId}/observation?${query.toString()}`)
}

export async function submitMove(matchId: string, playerId: string, move: Move): Promise<MatchView> {
  return request<MatchView>(`/api/match/${matchId}/move`, {
    method: 'POST',
    body: JSON.stringify({ player_id: playerId, move })
  })
}

export async function fetchEvents(matchId: string): Promise<MatchEvent[]> {
  return request<MatchEvent[]>(`/api/match/${matchId}/events`)
}
