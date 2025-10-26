import time
from typing import Any, Literal

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from prompt_utils import build_messages
from utils import (
    is_codenames,
    is_colonel_blotto,
    is_three_player_ipd,
    remove_trailing_spaces_keep_blank_lines,
)

GENERATION_PARAMS = {
    "Default": {
        "temperature": 1.0,
        "top_p": 1.0,
    },
    "Codenames": {
        "temperature": 1.0,
        "top_p": 1.0,
    },
    "ColonelBlotto": {
        "temperature": 1.0,
        "top_p": 1.0,
    },
    "ThreePlayerIPD": {
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
        },
    },
}


class Action(BaseModel):
    action: str | None = None
    action_parsing_failed: bool = False


class AgentResponse(BaseModel):
    prompt: str | None = None
    completion: str
    reasoning: str | None = None
    action: Action


class In2AIMindGamesAgent:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def __call__(self, observation: str) -> AgentResponse:
        messages = self._build_messages(observation)
        game_name: Literal["ThreePlayerIPD", "Codenames", "ColonelBlotto", "Default"] = (
            _extract_game_game_guideline(observation)
        )
        generation_params = GENERATION_PARAMS.get(game_name, GENERATION_PARAMS["Default"])
        response: ChatCompletion = self._run_request(messages, generation_params)
        completion = response.choices[0].message.content
        reasoning: str | None = getattr(response.choices[0].message, "reasoning_content", None)
        return AgentResponse(
            prompt=build_prompt_from_messages(messages),
            completion=completion,
            reasoning=reasoning,
            action=self._parse_action(completion),
        )

    def _run_request(
        self,
        messages: list[dict[str, str]],
        generation_params: dict[str, Any],
    ) -> ChatCompletion:
        last_exception: Exception | None = None

        for attempt_index in range(3):
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **generation_params,
                )
                if (
                    response is None
                    or len(response.choices) == 0
                    or response.choices[0].message is None
                    or response.choices[0].message.content is None
                ):
                    # Treat invalid/empty responses as retryable errors
                    last_exception = ValueError("No completion found in the response")
                else:
                    return response
            except Exception as exc:
                # Retry on internal/transport/API errors
                last_exception = exc
            # Sleep before next retry if we have remaining attempts
            if attempt_index < 2:
                time.sleep(1)
        # Exhausted retries
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Failed to get a valid completion after retries")

    def _build_messages(self, observation: str) -> list[dict[str, str]]:
        return build_messages(observation)

    def _parse_action(self, completion: str) -> Action:
        return Action(action=completion.strip(), action_parsing_failed=False)


def build_prompt_from_messages(messages: list[dict[str, str]]) -> str:
    prompt = ""
    for message in messages:
        prompt += "#" * 15 + message["role"].upper() + "#" * 15 + "\n"
        prompt += message["content"] + "\n"
    return prompt.strip()


def _extract_game_game_guideline(
    observation: str,
) -> Literal["ThreePlayerIPD", "Codenames", "ColonelBlotto", "Default"]:
    observation = remove_trailing_spaces_keep_blank_lines(observation)
    if is_colonel_blotto(observation):
        return "ColonelBlotto"
    if is_codenames(observation):
        return "Codenames"
    if is_three_player_ipd(observation):
        return "ThreePlayerIPD"
    return "Default"
