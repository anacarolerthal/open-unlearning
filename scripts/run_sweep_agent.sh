#!/usr/bin/env bash
set -euo pipefail

# Focused confirmation sweeps: ten fresh seeds (48--57) per model/method.
uv run wandb agent juanbelieni-lab/continual-open-unlearning/urzsn3pr
uv run wandb agent juanbelieni-lab/continual-open-unlearning/80g9eitg
uv run wandb agent juanbelieni-lab/continual-open-unlearning/do7lnws6
uv run wandb agent juanbelieni-lab/continual-open-unlearning/rq0y72to
