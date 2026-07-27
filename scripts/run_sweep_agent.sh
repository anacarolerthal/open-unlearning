#!/bin/bash
set -euo pipefail

# Register a sweep once yourself (prints an entity/project/sweep_id):
#   wandb sweep sweeps/npo_tofu_1b.yaml
#   wandb sweep sweeps/unlearn_head_tofu_1b.yaml
#
# Usage: bash scripts/run_sweep_agent.sh <entity/project/sweep_id>

sweep_id="${1:?usage: $0 <entity/project/sweep_id>}"
wandb agent "$sweep_id"
