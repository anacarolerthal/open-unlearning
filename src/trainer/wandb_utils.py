import os
import re


def log_wandb_artifact(path, name, artifact_type):
    import wandb

    if wandb.run is None or getattr(wandb.run, "disabled", False):
        return

    name = f"{wandb.run.id}-{name}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    artifact = wandb.Artifact(name=name, type=artifact_type)
    if os.path.isdir(path):
        artifact.add_dir(path)
    else:
        artifact.add_file(path)
    logged_artifact = wandb.run.log_artifact(artifact)
    if not getattr(wandb.run, "offline", False):
        logged_artifact.wait()
