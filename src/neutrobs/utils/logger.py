import wandb
import torch

def build_metric_key(metric):
    parts = []

    if metric.namespace:
        parts.append(metric.namespace)

    # optional layer grouping
    if "layer" in metric.tags:
        parts.append(metric.tags["layer"])

    parts.append(metric.name)

    return "/".join(parts)

class WandBLogger:
    def __init__(self, project="neutrobs", run_name=None, config=None):
        wandb.init(
            project=project, 
            name=run_name,
            config=config    
        )

    def log(self, metric):
        key = build_metric_key(metric)

        if metric.kind == "scalar":
            wandb.log(
                {key: metric.value},
                step=metric.step
            )

        elif metric.kind == "histogram":
            value = metric.value

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            value = value.reshape(-1)

            wandb.log(
                {key: wandb.Histogram(value)},
                step=metric.step
            )

        elif metric.kind == "image":
            wandb.log(
                {key: wandb.Image(metric.value.detach().cpu().numpy() if isinstance(metric.value, torch.Tensor) else metric.value)},
                step=metric.step
            )

        elif metric.kind == "heatmap":
            wandb.log(
                {key: wandb.Image(metric.value.detach().cpu().numpy() if isinstance(metric.value, torch.Tensor) else metric.value)}, 
                step=metric.step
            )