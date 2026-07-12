from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

from .config import WANDB_ENTITY, WANDB_PROJECT


def safe_wandb_init(run_name: str, group: str, job_type: str, config: dict | None = None):
    import wandb

    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        group=group,
        job_type=job_type,
        config=config or {},
    )


def log_dataframe_as_artifact(run, df: pd.DataFrame, name: str, artifact_type: str, path: str | Path):
    import wandb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    return artifact


def save_and_log_model(run, model, model_name: str, model_path: str | Path, metadata: dict | None = None, aliases=None):
    import wandb

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = metadata or {}
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifact = wandb.Artifact(model_name, type="model", metadata=metadata)
    artifact.add_file(str(model_path))
    artifact.add_file(str(metadata_path))
    run.log_artifact(artifact, aliases=aliases or ["latest"])
    return artifact


def download_model_artifact(artifact_name: str, alias: str = "latest", entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT):
    import wandb

    run = wandb.init(entity=entity, project=project, job_type="inference")
    artifact_ref = f"{entity}/{project}/{artifact_name}:{alias}"
    artifact = run.use_artifact(artifact_ref, type="model")
    path = Path(artifact.download())
    return run, path
