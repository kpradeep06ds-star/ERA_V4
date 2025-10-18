# log.py
import csv, json, time
from pathlib import Path

class MetricsLogger:
    def __init__(self, out_dir: str,
                 csv_name: str = "train_log.csv",
                 jsonl_name: str = "train_log.jsonl"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / csv_name
        self.jsonl_path = self.out_dir / jsonl_name
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def log_epoch(self, *, epoch: int, train_loss: float, val_loss: float,
                  acc1: float, acc5: float, lr: float, extra: dict | None = None):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "time": ts,
            "epoch": epoch,
            "lr": lr,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "acc1": float(acc1),
            "acc5": float(acc5),
            **(extra or {}),
        }
        # CSV
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)
        # JSONL
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def log_iter(self, *, epoch: int, it: int, iters: int, loss: float, lr: float):
        # Optional per-iteration logging to JSONL only (keeps CSV compact)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "time": ts, "epoch": epoch, "iter": it, "iters": iters,
            "loss": float(loss), "lr": float(lr)
        }
        with (self.out_dir / "iter_log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
