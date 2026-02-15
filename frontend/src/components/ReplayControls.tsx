interface ReplayControlsProps {
  replayTurn: number
  maxTurn: number
  isLive: boolean
  onPrev: () => void
  onNext: () => void
  onLive: () => void
}

export default function ReplayControls({ replayTurn, maxTurn, isLive, onPrev, onNext, onLive }: ReplayControlsProps) {
  return (
    <section className="replay-bar card">
      <div>
        <strong>{isLive ? 'Live' : 'Replay'}</strong>
        <span className="muted"> Turn {replayTurn} / {maxTurn}</span>
      </div>
      <div className="replay-actions">
        <button onClick={onPrev} disabled={replayTurn <= 0}>
          Back
        </button>
        <button onClick={onNext} disabled={replayTurn >= maxTurn}>
          Forward
        </button>
        <button onClick={onLive} disabled={isLive}>
          Jump To Live
        </button>
      </div>
    </section>
  )
}
