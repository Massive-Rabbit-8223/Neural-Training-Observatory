import duckdb
import json
import atexit
from typing import List
from neutrobs.utils.datatypes import Metric
import torch

class InMemoryStore:
    def __init__(self):
        self.metrics: List[Metric] = []

    def log(self, metric: Metric):
        self.metrics.append(metric)

    def query(self, name=None):
        return [m for m in self.metrics if name is None or m.name == name]

def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        else:
            raise ValueError(f"Non-scalar tensor: {value.shape}")
    return value

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
        value = _to_scalar(metric.value)

        self.buffer.append((
            metric.name,
            value,
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