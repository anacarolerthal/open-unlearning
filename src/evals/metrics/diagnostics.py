import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals


@unlearning_metric(name="lora_diff_diagnostics")
def lora_diff_diagnostics(model, **kwargs):
    diagnose_batch = getattr(model, "lora_diff_diagnostic_batch", None)
    if not callable(diagnose_batch):
        raise TypeError("lora_diff_diagnostics requires the LoraDiff trainer")

    dataloader = DataLoader(
        kwargs["data"],
        batch_size=kwargs["batch_size"],
        collate_fn=kwargs["collators"],
    )
    value_by_index = run_batchwise_evals(
        model,
        dataloader,
        lambda model, batch: diagnose_batch(batch),
        {},
        "Calculating LoRA difference diagnostics",
    )

    stat_names = next(iter(value_by_index.values())).keys()
    summary = {
        name: float(np.mean([values[name] for values in value_by_index.values()]))
        for name in stat_names
    }
    return {
        "agg_value": summary["correction_rms"],
        "summary": summary,
        "value_by_index": value_by_index,
    }
