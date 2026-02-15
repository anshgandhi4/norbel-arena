"""Codenames game implementation."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Mapping, Sequence

from framework.game import Game, LegalMovesSpec
from framework.result import MatchResult, TerminationReason

from .codenames_moves import EndTurn, GiveClue, Guess, MoveType, move_from_dict
from .codenames_observation import CodenamesObservation
from .codenames_state import (
    CardType,
    CodenamesState,
    Phase,
    Role,
    Team,
    player_for,
    team_role_for_player,
)

DEFAULT_WORDS: tuple[str, ...] = (
    "apple",
    "anchor",
    "angel",
    "arm",
    "back",
    "ball",
    "bank",
    "bar",
    "beach",
    "belt",
    "berry",
    "bird",
    "block",
    "board",
    "bolt",
    "boot",
    "bottle",
    "bow",
    "box",
    "bridge",
    "brush",
    "button",
    "cable",
    "camera",
    "camp",
    "cap",
    "card",
    "castle",
    "cell",
    "center",
    "chair",
    "change",
    "charge",
    "check",
    "circle",
    "clock",
    "cloud",
    "coat",
    "code",
    "coin",
    "cold",
    "cook",
    "copper",
    "cover",
    "crane",
    "crown",
    "dance",
    "date",
    "deck",
    "diamond",
    "doctor",
    "dog",
    "draft",
    "dream",
    "drop",
    "eagle",
    "engine",
    "eye",
    "fan",
    "field",
    "file",
    "film",
    "fire",
    "fish",
    "flag",
    "floor",
    "forest",
    "fork",
    "glass",
    "glove",
    "gold",
    "grass",
    "green",
    "hammer",
    "heart",
    "hook",
    "horn",
    "horse",
    "ice",
    "jet",
    "key",
    "king",
    "knife",
    "lab",
    "laser",
    "lead",
    "lemon",
    "light",
    "line",
    "link",
    "lock",
    "log",
    "map",
    "match",
    "mine",
    "mint",
    "moon",
    "mouse",
    "needle",
    "net",
    "night",
    "note",
    "oil",
    "orange",
    "paper",
    "park",
    "pen",
    "piano",
    "pilot",
    "pipe",
    "plane",
    "plate",
    "pool",
    "port",
    "post",
    "queen",
    "ring",
    "river",
    "robot",
    "rock",
    "root",
    "rose",
    "row",
    "ruler",
    "salt",
    "scale",
    "school",
    "screen",
    "seal",
    "server",
    "shadow",
    "ship",
    "shoe",
    "shop",
    "shot",
    "signal",
    "silver",
    "smoke",
    "snow",
    "song",
    "space",
    "spring",
    "star",
    "stone",
    "stream",
    "string",
    "suit",
    "sun",
    "table",
    "tank",
    "teacher",
    "temple",
    "thread",
    "time",
    "tool",
    "tower",
    "track",
    "train",
    "tree",
    "trip",
    "tube",
    "turkey",
    "vase",
    "watch",
    "water",
    "wave",
    "web",
    "wheel",
    "window",
    "wing",
    "wire",
    "wolf",
    "yard",
)

SAMPLE_CLUES: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "echo", "vector", "orbit", "spectrum")
FIXED_BOARD_SIZE = 25


class CodenamesGame(Game[CodenamesState, Any, CodenamesObservation]):
    """Two-team Codenames game with spymaster + operative roles."""

    game_name = "codenames"

    def __init__(self, default_config: dict[str, Any] | None = None):
        self.default_config = default_config or {}

    def new_game(self, seed: int, config: dict[str, Any] | None = None) -> CodenamesState:
        """Create a deterministic initial Codenames state."""
        cfg = dict(self.default_config)
        cfg.update(config or {})
        rng = random.Random(seed)

        board_size = FIXED_BOARD_SIZE

        words_source = tuple(cfg.get("word_list", DEFAULT_WORDS))
        if len(words_source) < board_size:
            raise ValueError(f"word_list must contain at least {board_size} words.")
        board_words = tuple(rng.sample(words_source, board_size))

        starting_team = self._parse_starting_team(cfg.get("starting_team"), rng)
        assassin_count = int(cfg.get("assassin_count", 1))
        red_count, blue_count, neutral_count = self._compute_assignment_counts(
            board_size=board_size,
            assassin_count=assassin_count,
            starting_team=starting_team,
            config=cfg,
        )

        assignments = (
            [CardType.RED] * red_count
            + [CardType.BLUE] * blue_count
            + [CardType.NEUTRAL] * neutral_count
            + [CardType.ASSASSIN] * assassin_count
        )
        rng.shuffle(assignments)

        return CodenamesState(
            seed=seed,
            board_words=board_words,
            assignments=tuple(assignments),
            revealed=tuple(False for _ in range(board_size)),
            turn_team=starting_team,
            phase=Phase.SPYMASTER_CLUE,
            current_clue=None,
            guesses_remaining=0,
            winner=None,
            termination_reason=None,
            turn_index=0,
            last_move=None,
            red_target_count=red_count,
            blue_target_count=blue_count,
        )

    def player_ids(self, state: CodenamesState) -> Sequence[str]:
        """Return canonical Codenames seats."""
        return (
            player_for(Team.RED, Role.SPYMASTER),
            player_for(Team.RED, Role.OPERATIVE),
            player_for(Team.BLUE, Role.SPYMASTER),
            player_for(Team.BLUE, Role.OPERATIVE),
        )

    def role_for_player(self, state: CodenamesState, player_id: str) -> str | None:
        """Return role for player ID."""
        _, role = team_role_for_player(player_id)
        return role.value

    def current_player(self, state: CodenamesState) -> str:
        """Return current player seat from turn team and phase."""
        if state.phase is Phase.SPYMASTER_CLUE:
            return player_for(state.turn_team, Role.SPYMASTER)
        return player_for(state.turn_team, Role.OPERATIVE)

    def legal_moves(self, state: CodenamesState, player_id: str) -> LegalMovesSpec:
        """Return legal move list/spec for current player."""
        if self.is_terminal(state):
            return []

        if player_id != self.current_player(state):
            return []

        if state.phase is Phase.SPYMASTER_CLUE:
            max_count = max(state.team_words_remaining(state.turn_team), 1)
            sample_moves = [
                GiveClue(clue=clue, count=count)
                for clue in SAMPLE_CLUES
                for count in range(0, min(3, max_count) + 1)
            ]
            return {
                "enumerable": False,
                "phase": state.phase.value,
                "template": {"type": MoveType.GIVE_CLUE.value, "clue": "string", "count": "int>=0"},
                "sample_moves": sample_moves,
            }

        moves = [Guess(index=index) for index, revealed in enumerate(state.revealed) if not revealed]
        moves.append(EndTurn())
        return moves

    def is_legal(self, state: CodenamesState, player_id: str, move: Any) -> tuple[bool, str | None]:
        """Validate legality of move for current state and player."""
        if self.is_terminal(state):
            return False, "Game is already terminal."
        if player_id != self.current_player(state):
            return False, f"It is not {player_id}'s turn."

        team, role = team_role_for_player(player_id)
        if team is not state.turn_team:
            return False, "Wrong team for current turn."

        if state.phase is Phase.SPYMASTER_CLUE:
            if role is not Role.SPYMASTER:
                return False, "Only spymaster can give clue in SPYMASTER_CLUE phase."
            if not isinstance(move, GiveClue):
                return False, "Expected GiveClue in SPYMASTER_CLUE phase."
            if not move.clue.strip():
                return False, "Clue cannot be empty."
            if move.count < 0:
                return False, "Clue count must be >= 0."
            return True, None

        if role is not Role.OPERATIVE:
            return False, "Only operative can act in OPERATIVE_GUESSING phase."
        if isinstance(move, EndTurn):
            return True, None
        if isinstance(move, Guess):
            if move.index < 0 or move.index >= len(state.board_words):
                return False, "Guess index out of range."
            if state.revealed[move.index]:
                return False, "Card already revealed."
            return True, None
        return False, "Expected Guess or EndTurn in OPERATIVE_GUESSING phase."

    def apply_move(self, state: CodenamesState, player_id: str, move: Any) -> CodenamesState:
        """Apply a legal move and return next immutable state."""
        legal, reason = self.is_legal(state, player_id, move)
        if not legal:
            raise ValueError(f"Illegal move: {reason}")

        if isinstance(move, GiveClue):
            return replace(
                state,
                phase=Phase.OPERATIVE_GUESSING,
                current_clue=(move.clue, move.count),
                guesses_remaining=move.count + 1,
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )

        if isinstance(move, EndTurn):
            return replace(
                state,
                turn_team=self._other_team(state.turn_team),
                phase=Phase.SPYMASTER_CLUE,
                current_clue=None,
                guesses_remaining=0,
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )

        if isinstance(move, Guess):
            revealed = list(state.revealed)
            revealed[move.index] = True
            next_state = replace(
                state,
                revealed=tuple(revealed),
                turn_index=state.turn_index + 1,
                last_move=move.to_dict(),
            )
            guessed_type = state.assignments[move.index]

            if guessed_type is CardType.ASSASSIN:
                return replace(
                    next_state,
                    winner=self._other_team(state.turn_team),
                    termination_reason="assassin",
                )

            winner = self._winner_if_targets_revealed(next_state)
            if winner is not None:
                return replace(
                    next_state,
                    winner=winner,
                    termination_reason="all_words_revealed",
                )

            if guessed_type is self._team_card_type(state.turn_team):
                remaining_guesses = state.guesses_remaining - 1
                if remaining_guesses <= 0:
                    return replace(
                        next_state,
                        turn_team=self._other_team(state.turn_team),
                        phase=Phase.SPYMASTER_CLUE,
                        current_clue=None,
                        guesses_remaining=0,
                    )
                return replace(next_state, guesses_remaining=remaining_guesses)

            return replace(
                next_state,
                turn_team=self._other_team(state.turn_team),
                phase=Phase.SPYMASTER_CLUE,
                current_clue=None,
                guesses_remaining=0,
            )

        raise ValueError(f"Unsupported move type: {type(move)!r}")

    def is_terminal(self, state: CodenamesState) -> bool:
        """Return whether the state has reached a terminal condition."""
        if state.winner is not None:
            return True
        return state.termination_reason == "draw"

    def outcome(self, state: CodenamesState) -> MatchResult:
        """Build structured outcome from terminal (or adjudicated) state."""
        red_revealed = state.red_target_count - state.team_words_remaining(Team.RED)
        blue_revealed = state.blue_target_count - state.team_words_remaining(Team.BLUE)
        scores = {
            Team.RED.value: float(red_revealed),
            Team.BLUE.value: float(blue_revealed),
        }

        if state.winner is not None:
            reason = TerminationReason.NORMAL_WIN
        else:
            reason = TerminationReason.DRAW

        return MatchResult(
            game_id="",
            game_name=self.game_name,
            seed=state.seed,
            winner=state.winner.value if state.winner is not None else None,
            termination_reason=reason,
            scores=scores,
            turns=state.turn_index,
            stats={
                "remaining_targets": {
                    Team.RED.value: state.team_words_remaining(Team.RED),
                    Team.BLUE.value: state.team_words_remaining(Team.BLUE),
                },
                "revealed_counts": state.revealed_counts(),
            },
            details=state.termination_reason,
            final_state_digest=state.state_digest(),
            event_count=0,
            log_path=None,
        )

    def observation(self, state: CodenamesState, player_id: str) -> CodenamesObservation:
        """Return partial observation for seat."""
        team, role = team_role_for_player(player_id)
        revealed_colors = tuple(
            assignment if revealed else None
            for assignment, revealed in zip(state.assignments, state.revealed, strict=True)
        )
        assignments = state.assignments if role is Role.SPYMASTER else None
        return CodenamesObservation(
            player_id=player_id,
            team=team,
            role=role,
            board_words=state.board_words,
            revealed=state.revealed,
            revealed_colors=revealed_colors,
            turn_team=state.turn_team,
            phase=state.phase,
            current_clue=state.current_clue,
            guesses_remaining=state.guesses_remaining,
            remaining_targets={
                Team.RED.value: state.team_words_remaining(Team.RED),
                Team.BLUE.value: state.team_words_remaining(Team.BLUE),
            },
            turn_index=state.turn_index,
            last_move=state.last_move,
            assignments=assignments,
        )

    def render(self, state: CodenamesState, player_id: str | None = None) -> str:
        """Render board for debugging."""
        show_assignments = False
        if player_id is not None:
            _, role = team_role_for_player(player_id)
            show_assignments = role is Role.SPYMASTER

        size = len(state.board_words)
        cols = int(math.sqrt(size))
        if cols * cols != size:
            cols = min(5, size)
        lines: list[str] = []
        for index, word in enumerate(state.board_words):
            if state.revealed[index]:
                token = f"{word}:{state.assignments[index].value}"
            elif show_assignments:
                token = f"{word}:{state.assignments[index].value.lower()}"
            else:
                token = word
            lines.append(f"{index:02d}:{token}")

        rows = [" | ".join(lines[row : row + cols]) for row in range(0, len(lines), cols)]
        header = (
            f"turn_team={state.turn_team.value} phase={state.phase.value} "
            f"clue={state.current_clue} guesses_remaining={state.guesses_remaining} winner={state.winner}"
        )
        return header + "\n" + "\n".join(rows)

    def parse_move(self, data: Mapping[str, Any]) -> Any:
        """Parse move payload into move object."""
        return move_from_dict(data)

    def forfeit_winner(self, state: CodenamesState, offending_player_id: str, reason: str) -> str | None:
        """Opponent team wins on runner-enforced forfeit."""
        team, _ = team_role_for_player(offending_player_id)
        return self._other_team(team).value

    def _parse_starting_team(self, raw: Any, rng: random.Random) -> Team:
        if raw is None:
            return rng.choice([Team.RED, Team.BLUE])
        if isinstance(raw, Team):
            return raw
        value = str(raw).upper()
        if value == Team.RED.value:
            return Team.RED
        if value == Team.BLUE.value:
            return Team.BLUE
        raise ValueError(f"Invalid starting_team: {raw!r}")

    def _compute_assignment_counts(
        self,
        *,
        board_size: int,
        assassin_count: int,
        starting_team: Team,
        config: Mapping[str, Any],
    ) -> tuple[int, int, int]:
        if assassin_count < 1:
            raise ValueError("assassin_count must be >= 1.")

        explicit_keys = ("red_count", "blue_count", "neutral_count")
        if all(key in config for key in explicit_keys):
            red_count = int(config["red_count"])
            blue_count = int(config["blue_count"])
            neutral_count = int(config["neutral_count"])
        else:
            neutral_count = max(1, board_size // 3)
            remaining = board_size - assassin_count - neutral_count
            if remaining < 2:
                raise ValueError("Board too small for team assignments.")
            red_count = remaining // 2
            blue_count = remaining - red_count
            if red_count == blue_count and neutral_count > 0:
                neutral_count -= 1
                if starting_team is Team.RED:
                    red_count += 1
                else:
                    blue_count += 1

        total = red_count + blue_count + neutral_count + assassin_count
        if total != board_size:
            raise ValueError(
                "Invalid assignment counts: "
                f"red={red_count}, blue={blue_count}, neutral={neutral_count}, assassin={assassin_count}, "
                f"board_size={board_size}"
            )
        return red_count, blue_count, neutral_count

    def _other_team(self, team: Team) -> Team:
        return Team.BLUE if team is Team.RED else Team.RED

    def _team_card_type(self, team: Team) -> CardType:
        return CardType.RED if team is Team.RED else CardType.BLUE

    def _winner_if_targets_revealed(self, state: CodenamesState) -> Team | None:
        if state.team_words_remaining(Team.RED) == 0:
            return Team.RED
        if state.team_words_remaining(Team.BLUE) == 0:
            return Team.BLUE
        return None
