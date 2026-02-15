const PLAYER_LABELS: Record<string, string> = {
  RED_SPYMASTER: 'Red Spymaster',
  RED_OPERATIVE: 'Red Operative',
  BLUE_SPYMASTER: 'Blue Spymaster',
  BLUE_OPERATIVE: 'Blue Operative',
  PSYCHIC: 'Psychic',
  GUESSER_ONE: 'Guesser One',
  GUESSER_TWO: 'Guesser Two'
}

function titleCaseWords(value: string): string {
  return value
    .trim()
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ')
}

export function formatPlayerId(playerId: string): string {
  const normalized = playerId.trim().toUpperCase().replace(/\s+/g, '_')
  if (normalized in PLAYER_LABELS) {
    return PLAYER_LABELS[normalized]
  }
  return titleCaseWords(playerId)
}

export function replacePlayerIdsInText(text: string): string {
  let next = text
  for (const [raw, pretty] of Object.entries(PLAYER_LABELS)) {
    next = next.replace(new RegExp(`\\b${raw}\\b`, 'g'), pretty)
  }
  return next
}
