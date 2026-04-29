# neutrobs/utils/tensor_processor.py

import torch
from neutrobs.utils.datatypes import Metric


class TensorProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, metric: Metric):
        tensor = metric.value.detach()

        if metric.semantics == "image":
            return self._process_image(metric, tensor)

        elif metric.semantics == "matrix":
            return self._process_matrix(metric, tensor)

        else:
            return self._process_distribution(metric, tensor)

    # -------------------------
    # Distribution
    # -------------------------
    def _process_distribution(self, metric, tensor):
        t = tensor.flatten()

        summaries = self._summaries(metric, t)

        hist = Metric(
            name=f"{metric.name}_hist",
            value=t.cpu(),
            step=metric.step,
            run_id=metric.run_id,
            tags=metric.tags,
            kind="histogram",
            namespace=metric.namespace,
            log_to_store=False,
            log_to_logger=True
        )

        return [hist] + summaries

    # -------------------------
    # Image
    # -------------------------
    def _process_image(self, metric, tensor):
        return [
            Metric(
                name=f"{metric.name}_img",
                value=tensor.detach().cpu(),
                step=metric.step,
                run_id=metric.run_id,
                tags=metric.tags,
                kind="image",
                namespace=metric.namespace,
                log_to_store=False,
                log_to_logger=True
            )
        ]

    # -------------------------
    # Matrix / heatmap
    # -------------------------
    def _process_matrix(self, metric, tensor):
        t = tensor.detach()

        summaries = self._summaries(metric, t.flatten())

        heatmap = Metric(
            name=f"{metric.name}_heatmap",
            value=t.cpu(),
            step=metric.step,
            run_id=metric.run_id,
            tags=metric.tags,
            kind="heatmap",
            namespace=metric.namespace,
            log_to_store=False,
            log_to_logger=True
        )

        return [heatmap] + summaries

    # -------------------------
    # Shared summaries
    # -------------------------
    def _summaries(self, metric, t):
        summaries = []

        def add(name, value):
            summaries.append(
                Metric(
                    name=f"{metric.name}_{name}",
                    value=value,
                    step=metric.step,
                    run_id=metric.run_id,
                    tags=metric.tags,
                    kind="scalar",
                    namespace=metric.namespace,
                    log_to_store=True,
                    log_to_logger=True
                )
            )

        if self.config.mean:
            add("mean", t.mean().item())

        if self.config.std:
            add("std", t.std().item())

        if self.config.min:
            add("min", t.min().item())

        if self.config.max:
            add("max", t.max().item())

        if self.config.p50:
            add("p50", t.quantile(0.5).item())

        if self.config.p90:
            add("p90", t.quantile(0.9).item())

        if self.config.p99:
            add("p99", t.quantile(0.99).item())

        if self.config.sparsity:
            add("sparsity", (t == 0).float().mean().item())

        return summaries