import re
from typing import Any, Literal


def remove_trailing_spaces_keep_blank_lines(text: str) -> str:
    """Remove trailing spaces/tabs at end of lines without collapsing blank lines.

    - Preserves multiple consecutive newlines
    - Only trims spaces and tabs that appear right before a newline or at EOF
    """
    # Remove spaces/tabs immediately before a newline (CRLF or LF)
    cleaned = re.sub(r"[ \t]+(?=\r?\n)", "", text)
    # Remove trailing spaces/tabs at very end of string (no newline at EOF)
    cleaned = re.sub(r"[ \t]+$", "", cleaned)
    return cleaned.strip()


def is_colonel_blotto(observation: str) -> bool:
    pattern = re.compile(
        r"(You are\s+.+?\s+in a game of ColonelBlotto\." r"|COLONEL\s+BLOTTO)",
        re.IGNORECASE | re.DOTALL,
    )
    return bool(pattern.search(observation))


def _extract_colonel_blotto_role(observation: str) -> str | None:
    m = re.search(
        r"You are\s+(.*?)\s+in a game of ColonelBlotto\.",
        observation,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _extract_colonel_blotto_units_from_board(observation: str) -> int | None:
    """Return the most recent 'Units to allocate' value from the board."""
    matches = list(
        re.finditer(r"Units to allocate:\s*(\d+)", observation, flags=re.IGNORECASE),
    )
    if not matches:
        return None
    m = matches[-1]
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_colonel_blotto_fields_from_board(
    observation: str,
) -> list[str] | None:
    """Return the most recent 'Available fields' list from the board."""
    pattern = re.compile(
        r"""
        Available\ fields:\s*
        ([A-Za-z](?:\s*,\s*[A-Za-z])*)
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    matches = list(pattern.finditer(observation))
    if not matches:
        return None
    m = matches[-1]
    fields_str = m.group(1)
    return [f.strip().upper() for f in fields_str.split(",") if f.strip()]


def _extract_colonel_blotto_round_info(
    observation: str,
) -> tuple[int | None, int | None]:
    """Return the most recent "Round X/Y" match from the observation."""
    matches = list(
        re.finditer(
            r"COLONEL\s+BLOTTO\s*-\s*Round\s*(\d+)\/(\d+)",
            observation,
            flags=re.IGNORECASE,
        ),
    )
    if not matches:
        return None, None
    last = matches[-1]
    try:
        return int(last.group(1)), int(last.group(2))
    except ValueError:
        return None, None


def extract_colonel_blotto_context(observation: str) -> dict[str, Any]:
    """Extract dynamic game details from the observation."""
    role = _extract_colonel_blotto_role(observation)
    num_units = _extract_colonel_blotto_units_from_board(observation)
    fields = _extract_colonel_blotto_fields_from_board(observation)
    current_round, total_rounds = _extract_colonel_blotto_round_info(observation)
    return {
        "role": role,
        "num_units": num_units,
        "fields": fields,
        "current_round": current_round,
        "total_rounds": total_rounds,
    }


def get_codenames_role(observation: str) -> Literal["Spymaster", "Operative"]:
    match = re.search(
        r"You are Player\s+\d+,\s+the\s+(Spymaster|Operative)\s+for",
        observation,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Codenames role not found in observation")
    return match.group(1)


def is_three_player_ipd(observation: str) -> bool:
    """Heuristically detect the 3-player Iterated Prisoner's Dilemma env.

    Primary signal is the initial prompt line. Fallbacks cover later-round boards.
    """
    primary = re.compile(
        r"You\s+are\s+Player\s+\d+\s+in\s+a\s+3-player\s+Iterated\s+Prisoner'?s\s+Dilemma",
        re.IGNORECASE,
    )
    if primary.search(observation):
        return True
    fallbacks = [
        re.compile(r"3-player\s+Iterated\s+Prisoner'?s\s+Dilemma", re.IGNORECASE),
        re.compile(r"\[\s*\d+\s+(cooperate|defect)\s*\]", re.IGNORECASE),
    ]
    return any(p.search(observation) for p in fallbacks)


def _ipd_extract_player_id(observation: str) -> int | None:
    m = re.search(
        r"You\s+are\s+Player\s+(\d+)\s+in\s+a\s+3-player\s+Iterated\s+Prisoner'?s\s+Dilemma",
        observation,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _ipd_extract_num_rounds(observation: str) -> int | None:
    m = re.search(r"The\s+match\s+lasts\s+(\d+)\s+rounds", observation, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _ipd_extract_conversation_turns_total(observation: str) -> int | None:
    # Bullet in initial prompt
    m = re.search(r"•\s*(\d+)\s+free-chat\s+turns", observation, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    # Board message
    m = re.search(
        r"converse\s+freely\s+for\s+the\s+next\s+(\d+)\s+rounds",
        observation,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _ipd_detect_phase(observation: str) -> Literal["conversation", "decision"]:
    obs = observation or ""
    # Find the LAST occurrence of each hint in the observation stream.
    dec_last = None
    for m in re.compile(
        r"^\[GAME\][^\n]*submit your decisions",
        re.IGNORECASE | re.MULTILINE,
    ).finditer(obs):
        dec_last = m.start()
    conv_last = None
    for m in re.compile(
        r"^\[GAME\][^\n]*You can converse freely",
        re.IGNORECASE | re.MULTILINE,
    ).finditer(obs):
        conv_last = m.start()

    if dec_last is not None or conv_last is not None:
        # If both exist, whichever appears later determines the current phase
        if dec_last is not None and (conv_last is None or dec_last > conv_last):
            return "decision"
        return "conversation"

    # Fallback heuristic near the end of the observation only
    tail = obs[-1000:]
    token_like = list(
        re.compile(r"\[\s*(\d+)\s+(cooperate|defect)\s*\]", re.IGNORECASE).finditer(tail),
    )
    return "decision" if len(token_like) >= 1 else "conversation"


def _ipd_extract_current_round(observation: str) -> int | None:
    obs = observation or ""
    pattern_start = re.compile(r"Starting\s+Round\s+(\d+)", re.IGNORECASE)
    pattern_results = re.compile(r"###\s*Round\s+(\d+)\s*-\s*Results", re.IGNORECASE)

    last_match = None
    for m in pattern_start.finditer(obs):
        if last_match is None or m.start() > last_match.start():
            last_match = m
    for m in pattern_results.finditer(obs):
        if last_match is None or m.start() > last_match.start():
            last_match = m

    if last_match is None:
        return None
    try:
        return int(last_match.group(1))
    except ValueError:
        return None


def _ipd_extract_payoffs(observation: str) -> dict[str, int | None]:
    def _grab(pattern: str) -> int | None:
        m = re.search(pattern, observation, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    cc = _grab(r"Both\s+cooperate\s*->\s*(\d+)")
    dd = _grab(r"Both\s+defect\s*->\s*(\d+)")
    t = _grab(r"You\s+defect,\s*they\s+cooperate\s*->\s*(\d+)")
    s = _grab(r"You\s+cooperate,\s*they\s+defect\s*->\s*(\d+)")
    # Map to conventional R, T, S, P names from env
    return {"R": cc, "P": dd, "T": t, "S": s}


def extract_three_player_ipd_context(observation: str) -> dict[str, Any]:
    player_id = _ipd_extract_player_id(observation)
    num_rounds = _ipd_extract_num_rounds(observation)
    conversation_turns_total = _ipd_extract_conversation_turns_total(observation)
    phase = _ipd_detect_phase(observation)
    current_round = _ipd_extract_current_round(observation)
    payoffs = _ipd_extract_payoffs(observation)

    opponent_ids: list[int] | None = None
    if player_id is not None and player_id in {0, 1, 2}:
        opponent_ids = sorted([pid for pid in (0, 1, 2) if pid != player_id])

    return {
        "player_id": player_id,
        "num_rounds": num_rounds,
        "conversation_turns_total": conversation_turns_total,
        "phase": phase,
        "current_round": current_round,
        "R": payoffs.get("R"),
        "T": payoffs.get("T"),
        "S": payoffs.get("S"),
        "P": payoffs.get("P"),
        "opponent_ids": opponent_ids,
    }


def is_codenames(observation: str) -> bool:
    pattern = re.compile(
        r"You are playing Codenames,?\s*a 2v2 word deduction game\.",
        re.IGNORECASE,
    )
    return bool(pattern.search(observation))


def extract_codenames_context(observation: str) -> dict[str, Any]:
    """Extract Codenames context: team color and last clue/number if present."""
    team: str | None = None
    last_clue_word: str | None = None
    last_clue_number: int | None = None

    # Team from role prompt line
    m_prompt_team = re.search(
        r"You are Player\s+\d+,\s+the\s+(Spymaster|Operative)\s+for\s+(Red|Blue)\s+team\.",
        observation,
        flags=re.IGNORECASE,
    )
    if m_prompt_team:
        team = m_prompt_team.group(2).capitalize()

    # Team and last clue from action description lines (pick the LAST match;
    # prefer user's team if available)
    submitted_pattern = re.compile(
        (
            r"Spymaster\s+of\s+(Red|Blue)\s+team,\s+Player\s+\d+,\s+submitted\s+\["
            r"(\w+)\s+(\d+)\]\."
        ),
        flags=re.IGNORECASE,
    )
    submitted_matches = list(submitted_pattern.finditer(observation))
    if submitted_matches:
        selected_match = None
        if team is not None:
            # Prefer the last match from the user's team
            for m in reversed(submitted_matches):
                submitted_team = m.group(1).capitalize()
                if submitted_team.lower() == team.lower():
                    selected_match = m
                    break
        # Fallback: use the very last match overall
        if selected_match is None:
            selected_match = submitted_matches[-1]

        selected_team = selected_match.group(1).capitalize()
        selected_word = selected_match.group(2).lower()
        try:
            selected_number = int(selected_match.group(3))
        except ValueError:
            selected_number = None

        # Fill outputs
        last_clue_word = selected_word
        last_clue_number = selected_number
        # If team was not known from the prompt, set it from the selected submission
        if team is None:
            team = selected_team

    return {
        "team": team,
        "last_clue_word": last_clue_word,
        "last_clue_number": last_clue_number,
    }
