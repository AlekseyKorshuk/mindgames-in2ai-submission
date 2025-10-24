import json
import logging
import multiprocessing as mp
import os
import signal
from datetime import UTC, datetime
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any, Literal

import click
import textarena as ta
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent import AgentResponse, In2AIMindGamesAgent

logging.basicConfig(level=logging.INFO)


class GameStep(BaseModel):
    player_id: int
    observation: str
    action: AgentResponse
    step_info: dict[str, Any]


class OnlineEnvironmentInfo(BaseModel):
    game_url: str
    env_id: str
    environment_id: int
    matched_env_name: str


class GameEvaluationLog(BaseModel):
    public_model_name: str
    public_model_description: str
    track: Literal["Social Detection", "Generalization"]
    small_category: bool
    start_time: datetime
    end_time: datetime | None = None
    rewards: dict[str, float] | None = None
    game_info: dict[str, Any] | None = None
    steps: list[GameStep] = Field(default_factory=list)
    online_environment_info: OnlineEnvironmentInfo | None = None


def parse_max_games_option(_: click.Context, __: click.Parameter, value: str | None) -> int | None:
    """
    Parse the --max-games-per-track option: accepts integer > 0 or 'None' for no limit.
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"none", ""}:
        return None
    try:
        num = int(value)
    except ValueError as err:
        raise click.BadParameter("must be an integer > 0 or 'None'") from err
    if num <= 0:
        raise click.BadParameter("must be an integer > 0 or 'None'")
    return num


@click.command()
@click.option(
    "--model-name",
    default="Qwen/Qwen3-8B",
    type=str,
    show_default=True,
    help="Model name for the agent.",
)
@click.option(
    "--base-url",
    default=None,
    type=str,
    required=False,
    help="Base URL for the agent.",
)
@click.option(
    "--api-key",
    default=None,
    type=str,
    required=False,
    help="API key for the agent.",
)
@click.option(
    "--public-model-name",
    default="In2AI/Baseline",
    type=str,
    show_default=True,
    help="Public model name for the agent.",
)
@click.option(
    "--public-model-description",
    default="",
    type=str,
    show_default=True,
    help="Model description for the agent.",
)
@click.option(
    "--tracks",
    default="Generalization",
    type=click.Choice(["Generalization", "Social Detection", "All"], case_sensitive=True),
    show_default=True,
    help="Tracks for the agent.",
)
@click.option(
    "--small-category",
    default=True,
    type=bool,
    show_default=True,
    help="Whether to participate in the small category.",
)
@click.option(
    "--logs-jsonl-directory",
    default="./in2ai-submission-stage-2-logs/online/",
    type=str,
    show_default=True,
    help="Path to the directory to save the logs.",
)
@click.option(
    "--max-games-per-track",
    default=None,
    type=str,
    show_default=True,
    help="Maximum games each track plays. Use 'None' for no limit.",
    callback=parse_max_games_option,
)
@click.option(
    "--team-hash",
    required=True,
    type=str,
    help="Team hash for online play (provided by TextArena).",
)
def main(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    public_model_name: str,
    public_model_description: str,
    tracks: Literal["Generalization", "Social Detection", "All"],
    small_category: bool,
    logs_jsonl_directory: str,
    max_games_per_track: int | None,
    team_hash: str,
) -> None:
    public_model_name_path = public_model_name.replace("/", "_").replace(":", "_").replace("-", "_")
    logs_directory_path = Path(logs_jsonl_directory) / f"{public_model_name_path}"
    logs_directory_path.mkdir(parents=True, exist_ok=True)

    # Resolve tracks to spawn workers for
    if tracks == "All":
        selected_tracks: list[Literal["Social Detection", "Generalization"]] = [
            "Generalization",
            "Social Detection",
        ]
    else:
        selected_tracks = [tracks]

    stop_event: MpEvent = mp.Event()

    processes: list[mp.Process] = []
    termination_signals_received: int = 0

    def handle_shutdown(signum: int, _frame: object | None) -> None:
        nonlocal termination_signals_received
        termination_signals_received += 1

        if termination_signals_received == 1:
            logging.info(
                "Received signal %s. Requesting graceful shutdown (finish current game).",
                signum,
            )
            stop_event.set()
            return

        logging.warning(
            "Received second termination signal %s. Forcing immediate exit.",
            signum,
        )
        # Best-effort: kill any still-alive child processes before exiting
        for p in processes:
            try:
                if p.is_alive():
                    p.kill()
            except Exception:
                logging.exception(
                    "Failed to kill child process %s during forced exit",
                    p.pid,
                )
        os._exit(1)

    # Register signal handlers in the main process
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Start worker processes
    for trk in selected_tracks:
        p = mp.Process(
            target=start_worker,
            name=f"worker-{trk.replace(' ', '_').lower()}",
            args=(
                model_name,
                base_url,
                api_key,
                public_model_name,
                public_model_description,
                small_category,
                trk,
                team_hash,
                logs_directory_path,
                stop_event,
                max_games_per_track,
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Wait for workers to complete. They will exit after finishing the current game
    # once a shutdown signal is received (stop_event is set).
    for p in processes:
        try:
            p.join()
        except KeyboardInterrupt:
            # Ensure event is set and continue waiting
            stop_event.set()
            p.join()


def start_worker(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    public_model_name: str,
    public_model_description: str,
    small_category: bool,
    track: Literal["Social Detection", "Generalization"],
    team_hash: str,
    logs_directory_path: Path,
    stop_event: MpEvent,
    max_games: int | None,
) -> None:
    logging.info("Worker started for track=%s", track)

    games_played = 0
    played_after_shutdown = 0

    while True:
        # Respect max games limit if set
        if max_games is not None and games_played >= max_games:
            logging.info("Max games reached (%s). Exiting worker for track=%s", max_games, track)
            break

        # After a shutdown request, allow exactly one last game to complete, then exit
        if stop_event.is_set():
            if played_after_shutdown >= 1:
                logging.info(
                    "Final game completed after shutdown; exiting worker for track=%s",
                    track,
                )
                break
            logging.info("Shutdown requested; will play one last game for track=%s", track)

        try:
            play_game(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                public_model_name=public_model_name,
                public_model_description=public_model_description,
                small_category=small_category,
                track=track,
                team_hash=team_hash,
                logs_directory_path=logs_directory_path,
            )
        except ValueError as ex:
            if "Model registration failed" in str(ex):
                logging.error("Model registration failed. Exiting worker for track=%s", track)
                break
            raise
        except Exception:
            logging.exception("Unhandled error in worker for track=%s; continuing.", track)
            continue

        games_played += 1
        logging.info("Games played: %s / %s", games_played, max_games)
        if stop_event.is_set():
            played_after_shutdown += 1


def play_game(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    public_model_name: str,
    public_model_description: str,
    small_category: bool,
    track: Literal["Social Detection", "Generalization"],
    team_hash: str,
    logs_directory_path: Path,
) -> None:
    agent = In2AIMindGamesAgent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    env = ta.make_mgc_online(
        track=track,
        model_name=public_model_name,
        model_description=public_model_description,
        team_hash=team_hash,
        agent=agent,
        small_category=small_category,
    )
    env.reset(num_players=1)

    game_evaluation_log = GameEvaluationLog(
        public_model_name=public_model_name,
        public_model_description=public_model_description,
        track=track,
        small_category=small_category,
        start_time=datetime.now(UTC),
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logs_jsonl_path = logs_directory_path / f"{timestamp}.json"

    done = False
    while not done:
        player_id, observation = env.get_observation()
        observation = _convert_observation_to_str(
            observation=observation,
            role_mapping=env.state.role_mapping,
        )
        logging.info("[Player %s] Observation:\n%s", player_id, observation)
        action: AgentResponse = agent(observation)
        logging.info("[Player %s] Reasoning:\n%s", player_id, action.completion)
        logging.info("[Player %s] Action:\n%s", player_id, action.action.action)
        done, step_info = env.step(action=action.action.action)
        logging.info("Step Info:\n%s", step_info)
        game_evaluation_log.steps.append(
            GameStep(
                player_id=player_id,
                observation=observation,
                action=action,
                step_info=step_info,
            ),
        )
    game_evaluation_log.online_environment_info = OnlineEnvironmentInfo(
        game_url=env.game_url,
        env_id=env.env_id,
        environment_id=env.environment_id,
        matched_env_name=env.matched_env_name,
    )

    rewards, game_info = env.close()
    logging.info("Rewards:\n%s", rewards)
    logging.info("Game Info:\n%s", game_info)
    game_evaluation_log.end_time = datetime.now(UTC)
    game_evaluation_log.rewards = rewards
    game_evaluation_log.game_info = game_info

    # append the game logs to the jsonl file
    with logs_jsonl_path.open("w") as f:
        json.dump(game_evaluation_log.model_dump(mode="json"), f)


def _convert_observation_to_str(
    observation: list[ta.Message] | str,
    role_mapping: dict[int, str],
) -> str:
    """Get the history conversation for the given player."""
    if isinstance(observation, str):
        return observation
    history = []
    for sender_id, message, _ in observation:
        if sender_id == ta.GAME_ID:
            sender_name = "GAME"
        else:
            sender_name = role_mapping.get(sender_id, f"Player {sender_id}")
        history.append(f"[{sender_name}] {message}")
    return "\n".join(history)


if __name__ == "__main__":
    load_dotenv()
    main()
