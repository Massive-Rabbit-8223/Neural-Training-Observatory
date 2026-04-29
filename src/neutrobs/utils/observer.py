from neutrobs.utils.datatypes import Metric
import torch
from torch import nn
from neutrobs.utils.router import MetricRouter

#----------------------
#   Observer Engine
#----------------------
class ObserverEngine:
    def __init__(self, modules, store=None, logger=None, tensor_processor=None, run_id="run_0"):
        """
        modules:    list of observer objects
        store:      memory store
        logger:     logger (e.g. wandb logger)
        """

        self.modules = modules
        self.router = MetricRouter(store, logger, tensor_processor)
        self.store = store
        self.logger = logger
        self.run_id = run_id

    def emit(self, event_type, **context):
        context["run_id"] = self.run_id
        context["event_type"] = event_type

        for module in self.modules:
            handler = getattr(module, f"on_{event_type}", None)
            if handler:
                metrics = handler(context)
                if metrics:
                    for m in metrics:
                        self.router.route(m)

    def close(self):
        for module in self.modules:
            if hasattr(module, "close"):
                module.close()

#----------------------
#   Observer Modules
#----------------------
class LossObserver:
    def on_forward_end(self, ctx):
        return [
            Metric(
                name="main",
                value=float(ctx["loss"].detach()),
                step=ctx["step"],
                run_id=ctx["run_id"],
                tags={},
                kind="scalar",
                namespace="loss"
            )
        ]


class GradNormObserver:
    def __init__(self, every_n_steps=50):
        self.every_n_steps = every_n_steps
        

    def on_backward_end(self, ctx):
        if ctx["step"] % self.every_n_steps != 0:
            return None
        else:
            model = ctx["get_model"]()
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.detach().pow(2).sum()

            total_norm = total.sqrt().item()

            return [
                Metric(
                    name="global",
                    value=total_norm,
                    step=ctx["step"],
                    run_id=ctx["run_id"],
                    tags={},
                    kind="scalar",
                    namespace="grad_norm"
                )
            ]
        

class GradObserver:
    def __init__(self, every_n_steps=50):
        self.every_n_steps = every_n_steps
        

    def on_backward_end(self, ctx):
        if ctx["step"] % self.every_n_steps != 0:
            return None
        else:
            model = ctx["get_model"]()

            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.flatten())

            grads = torch.cat(grads)

            return [
                Metric(
                    name="global",
                    value=grads.detach().cpu(),
                    step=ctx["step"],
                    run_id=ctx["run_id"],
                    tags={},
                    kind="tensor",
                    semantics="distribution",
                    namespace="grad"
                )
            ]
        

class ActivationStatsObserver:
    """
    Computes lightweight activation statistics per layer:
    - mean
    - std
    - min / max
    - sparsity
    - dead fraction (ReLU-like behavior)
    """

    def __init__(self, every_n_steps: int = 100, track_types=(nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
        self.every_n_steps = every_n_steps
        self.track_types = track_types

        self.hooks_registered = False
        self.handles = []

        # temporary storage (overwritten each forward pass)
        self.activations = {}

    # ----------------------------
    # Hook registration
    # ----------------------------
    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, self.track_types):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # detach immediately to avoid graph retention
            self.activations[name] = output.detach()
        return hook

    # ----------------------------
    # Event handler
    # ----------------------------
    def on_forward_end(self, ctx):
        step = ctx["step"]

        # sampling (critical for performance)
        if step % self.every_n_steps != 0:
            return None

        model = ctx["get_model"]()

        # lazy hook registration (only once)
        if not self.hooks_registered:
            self._register_hooks(model)
            self.hooks_registered = True
            return None

        metrics = []

        # compute stats
        for name, act in self.activations.items():
            metrics.append(
                Metric(
                    name="activation",
                    value=act,
                    step=step,
                    run_id=ctx["run_id"],
                    tags={"layer": name},
                    kind="tensor",
                    semantics="distribution",
                    namespace="activation"
                )
            )

        return metrics

    # ----------------------------
    # Cleanup (important for long runs)
    # ----------------------------
    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []