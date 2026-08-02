#!/bin/bash
set -euo pipefail

default_sweep_id="juanbelieni-lab/unlearn-head/x01clp5w"
sweep_id="${1:-${WANDB_SWEEP_ID:-$default_sweep_id}}"

# Run the agent in the same uv environment used by every sweep trial.
uv run wandb agent "$sweep_id"
