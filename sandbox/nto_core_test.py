from dataclasses import dataclass
from typing import Dict, Any, List
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import duckdb
import json
import atexit


#----------------------
#   Memory Store +
#   Query Engine
#----------------------
@dataclass
class Metric:
    name: str
    value: float
    step: int
    run_id: int
    tags: Dict[str, Any]

class InMemoryStore:
    def __init__(self):
        self.metrics: List[Metric] = []

    def log(self, metric: Metric):
        self.metrics.append(metric)

    def query(self, name=None):
        return [m for m in self.metrics if name is None or m.name == name]



class DuckDBStore:
    def __init__(self, db_path="metrics.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.buffer = []
        self._init_table()

        # Register automatic flush on program exit
        atexit.register(self.flush)

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                name TEXT,
                value DOUBLE,
                step INTEGER,
                run_id TEXT,
                tags TEXT
            )
        """)

    def log(self, metric):
        self.buffer.append((
            metric.name,
            metric.value,
            metric.step,
            metric.run_id,
            json.dumps(metric.tags)
        ))

        if len(self.buffer) >= 1000:
            self.flush()

    def flush(self):
        if self.buffer:
            self.conn.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                self.buffer
            )
            self.buffer = []

    def query(self, sql):
        return self.conn.execute(sql).fetchdf()

    def get_metric(self, name):
        return self.conn.execute("""
            SELECT * FROM metrics WHERE name = ?
        """, [name]).fetchdf()

#----------------------
#   Observer Engine
#----------------------
class ObserverEngine:
    def __init__(self, modules, store, run_id="run_0"):
        """
        modules:    observer objects
        store:      memory store
        """
        self.modules = modules
        self.store = store
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
                        self.store.log(m)

#----------------------
#   Example Observer 1
#----------------------
class LossObserver:
    def on_forward_end(self, ctx):
        return [
            Metric(
                name="loss",
                value=ctx["loss"].item(),
                step=ctx["step"],
                run_id=ctx["run_id"],
                tags={}
            )
        ]

#----------------------
#   Example Observer 2
#----------------------
class GradNormObserver:
    def on_backward_end(self, ctx):
        model = ctx["get_model"]()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum()

        total_norm = total.sqrt().item()

        return [
            Metric(
                name="grad_norm",
                value=total_norm,
                step=ctx["step"],
                run_id=ctx["run_id"],
                tags={}
            )
        ]
#----------------------
#   Adapter-Layer
#----------------------
def train(model, optimizer, data_loader, observer, epochs=1):
    step = 0

    for epoch in range(epochs):
        for batch in data_loader:
            x, y = batch

            observer.emit("batch_start", step=step, epoch=epoch)

            outputs = model(x)
            loss = ((outputs - y) ** 2).mean()

            observer.emit(
                "forward_end",
                step=step,
                loss=loss,
            )

            loss.backward()

            observer.emit(
                "backward_end",
                step=step,
                get_model=lambda: model
            )

            optimizer.step()
            optimizer.zero_grad()

            observer.emit("optimizer_step", step=step, epoch=epoch)

            step += 1
        
        observer.emit("epoch_end", epoch=epoch, step=step)

if __name__ == "__main__":

    # Dummy dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    # Model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Observatory setup
    #store = InMemoryStore()
    store = DuckDBStore("observatory_test.duckdb")

    observer = ObserverEngine(
        modules=[
            LossObserver(),
            GradNormObserver()
        ],
        store=store,
        run_id="demo_run"
    )

    # Train
    train(model, optimizer, loader, observer, epochs=2)
    store.flush()

    # Query stored metrics
    #df = store.query("""
    #    SELECT step, value
    #    FROM metrics
    #    WHERE name = 'loss'
    #    ORDER BY step
    #""")

    df = store.get_metric('loss')

    print(df)