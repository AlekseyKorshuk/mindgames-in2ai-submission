# Submission for MindGames: In2AI Team

## How to install (using uv)

1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2) Install and pin Python 3.12.10 with uv

```bash
uv python install 3.12.10
uv python pin 3.12.10
```

3) Create the virtual environment and install dependencies

```bash
uv sync --prerelease=allow
```

## How to Run

### SGLang Server

Firstly, to start the SGLang server with the model in a separate terminal / tmux session:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python3 -m sglang.launch_server \
    --model AlekseyKorshuk/mindgames-in2ai-submission \
    --reasoning-parser deepseek-r1 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --tp 1 \
    --dp-size 1
```

### **IMPORTANT**
> In case you observe that the context window of the model is too small for some of the games (most probably in rare cases for Codenames) - use RoPE scaling. But ONLY if actually required, do not enable by default, because it impacts model quality. Please, watch this issue, because other participants can be impacted by the same one for very long games.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python3 -m sglang.launch_server \
     ... \
     --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
```

### Connect to TextArena

Now you can connect to TextArena. It will serve the model for the given track in the loop until you stop it.
For graceful shutdown, you can press Ctrl+C **once** (or press Ctrl+C twice for immediate exit - active game will be terminated):
- If the game is playing right now, it will finish the game and then exit.
- If the game is not playing right now, it will exit immediately.

Do not forget to set:
- `model-name` as the same name as the one used in the SGLang server
- `base-url` as the same URL as the one used in the SGLang server
- `api-key` set to `EMPTY`, unless changed in the SGLang server
- `public-model-name` set to the actual submission name
- `tracks` keep `Generalization`
- `team-hash` set to the actual team hash

```bash
uv run python3 -m online_play_stage_2 \
    --model-name AlekseyKorshuk/mindgames-in2ai-submission \
    --base-url http://0.0.0.0:8000/v1 \
    --api-key EMPTY \
    --public-model-name In2AI \
    --tracks Generalization \
    --team-hash TEAM_HASH
```