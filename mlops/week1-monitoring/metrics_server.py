"""
ML Service Monitoring: Prometheus metrics + structured logging
Tracks: request rate, latency (p50/p95/p99), prediction distribution, errors.
"""

import logging
import random
import time
from contextlib import contextmanager

# ─── Structured Logging ──────────────────────────────────────────────────────

import json


class JSONLogger:
    """Structured JSON logging — makes logs queryable in ELK/CloudWatch."""

    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, message: str, **kwargs):
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs,
        }
        print(json.dumps(record))

    def info(self, msg: str, **kwargs):  self._log("INFO", msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log("WARNING", msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log("ERROR", msg, **kwargs)


logger = JSONLogger("ml-service")


# ─── Metrics Collector ────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Simple in-process metrics collector.
    In production: replace with prometheus_client library.
    Exposes: counters, histograms (for latency percentiles), gauges.
    """

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0, labels: dict = None):
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def set(self, name: str, value: float, labels: dict = None):
        self._gauges[self._key(name, labels)] = value

    def percentile(self, name: str, p: float, labels: dict = None) -> float:
        key = self._key(name, labels)
        data = sorted(self._histograms.get(key, [0]))
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    def _key(self, name: str, labels: dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def report(self) -> dict:
        latency_keys = [k for k in self._histograms if "latency" in k]
        report = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "latency_percentiles": {},
        }
        for key in latency_keys:
            report["latency_percentiles"][key] = {
                "p50": round(self.percentile(key.split("{")[0], 50), 2),
                "p95": round(self.percentile(key.split("{")[0], 95), 2),
                "p99": round(self.percentile(key.split("{")[0], 99), 2),
                "count": len(self._histograms[key]),
            }
        return report


metrics = MetricsCollector()


# ─── Instrumented Prediction Service ─────────────────────────────────────────

@contextmanager
def track_request(endpoint: str, model: str):
    """Context manager that records latency, success/error for any request."""
    start = time.perf_counter()
    labels = {"endpoint": endpoint, "model": model}
    metrics.inc("requests_total", labels=labels)
    try:
        yield
        metrics.inc("requests_success_total", labels=labels)
    except Exception as e:
        metrics.inc("requests_error_total", labels={**labels, "error": type(e).__name__})
        logger.error("Request failed", endpoint=endpoint, error=str(e))
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.observe("request_latency_ms", latency_ms, labels=labels)
        logger.info("Request completed", endpoint=endpoint, latency_ms=round(latency_ms, 2))


def predict(text: str, model: str = "classifier-v2") -> dict:
    """Instrumented prediction function."""
    with track_request("/predict", model):
        # Simulate variable latency
        time.sleep(random.uniform(0.01, 0.15))

        # Simulate occasional errors
        if random.random() < 0.05:
            raise RuntimeError("Model inference timeout")

        label = random.choice(["positive", "negative", "neutral"])
        confidence = random.uniform(0.6, 0.99)

        metrics.inc("predictions_total", labels={"label": label, "model": model})
        metrics.set("model_confidence_avg", confidence)

        logger.info(
            "Prediction made",
            model=model,
            label=label,
            confidence=round(confidence, 3),
            input_length=len(text),
        )

        return {"label": label, "confidence": round(confidence, 3)}


# ─── Prometheus Exposition Format ─────────────────────────────────────────────

def generate_prometheus_metrics() -> str:
    """
    In production: prometheus_client handles this automatically.
    This shows the text format Prometheus scrapes.
    """
    lines = []
    report = metrics.report()

    for key, value in report["counters"].items():
        lines.append(f"# TYPE {key.split('{')[0]} counter")
        lines.append(f"{key} {value}")

    for key, value in report["gauges"].items():
        lines.append(f"# TYPE {key} gauge")
        lines.append(f"{key} {value}")

    for key, percentiles in report["latency_percentiles"].items():
        metric_name = key.split("{")[0]
        lines.append(f"# TYPE {metric_name} summary")
        for quantile, value in [("0.5", percentiles["p50"]), ("0.95", percentiles["p95"]), ("0.99", percentiles["p99"])]:
            lines.append(f'{metric_name}{{quantile="{quantile}"}} {value}')
        lines.append(f"{metric_name}_count {percentiles['count']}")

    return "\n".join(lines)


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Simulating 50 Requests ===\n")

    success = error = 0
    for i in range(50):
        try:
            result = predict(f"Sample text number {i}", model="classifier-v2")
            success += 1
        except Exception:
            error += 1

    print(f"\n=== Results: {success} success, {error} errors ===\n")

    report = metrics.report()
    print("=== Metrics Report ===")
    print(json.dumps(report, indent=2))

    print("\n=== Prometheus Exposition Format ===")
    print(generate_prometheus_metrics()[:500] + "\n...")
