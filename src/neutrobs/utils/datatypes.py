from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Metric:
    name: str
    value: Any
    step: int
    run_id: str
    tags: Dict

    # type information
    kind: str = "scalar"        # scalar | tensor | histogram | image | heatmap
    semantics: str = None       # distribution | image | matrix | ...

    # routing policy
    log_to_store: bool = True
    log_to_logger: bool = True

    namespace: str = None       # e.g. "grad", "activation", "loss"


@dataclass
class TensorSummaryConfig:
    mean: bool = True
    std: bool = True
    min: bool = True
    max: bool = True
    p50: bool = False
    p90: bool = True
    p99: bool = True
    sparsity: bool = False