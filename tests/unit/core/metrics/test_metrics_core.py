import time

import pytest

from ember.core.metrics.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metrics,
    MetricsRegistry,
)


# Test Counter
def test_counter_init():
    c = Counter()
    assert c.get() == 0
    c = Counter(10)
    assert c.get() == 10


def test_counter_inc():
    c = Counter()
    c.inc()
    assert c.get() == 1
    c.inc(5)
    assert c.get() == 6


def test_counter_snapshot():
    c = Counter(5)
    assert c.snapshot() == 5
    c.inc()
    assert c.snapshot() == 6


# Test Gauge
def test_gauge_init():
    g = Gauge()
    assert g.get() == 0.0
    g = Gauge(10.5)
    assert g.get() == 10.5


def test_gauge_set():
    g = Gauge()
    g.set(5.5)
    assert g.get() == 5.5
    g.set(0.1)
    assert g.get() == 0.1


def test_gauge_snapshot():
    g = Gauge(7.7)
    assert g.snapshot() == 7.7
    g.set(1.2)
    assert g.snapshot() == 1.2


# Test Histogram
def test_histogram_init():
    h = Histogram()
    snap = h.snapshot()
    assert snap["count"] == 0
    assert snap["sum"] == 0.0
    assert snap["min"] == 0  # Default min is 0 if count is 0
    assert snap["max"] == 0  # Default max is 0 if count is 0
    assert snap["avg"] == 0.0
    assert len(snap["buckets"]) == len(Histogram.BUCKET_BOUNDARIES)
    assert all(b == 0 for b in snap["buckets"])


def test_histogram_record():
    h = Histogram()
    h.record(0.07)  # Bucket index 4 (0.05 < val <= 0.1)
    h.record(3.0)  # Bucket index 9 (2.5 < val <= 5)
    h.record(1500)  # Bucket index 17 (1000 < val <= inf)
    h.record(0.001)  # Bucket index 0 (val <= 0.005)

    snap = h.snapshot()
    assert snap["count"] == 4
    assert pytest.approx(snap["sum"]) == 0.07 + 3.0 + 1500 + 0.001
    assert snap["min"] == 0.001
    assert snap["max"] == 1500
    assert pytest.approx(snap["avg"]) == (0.07 + 3.0 + 1500 + 0.001) / 4

    assert snap["buckets"][0] == 1  # <= 0.005
    assert snap["buckets"][4] == 1  # <= 0.1
    assert snap["buckets"][9] == 1  # <= 5
    assert snap["buckets"][17] == 1  # <= inf (Index 17 for value 1500)
    # Check a few others are zero
    assert snap["buckets"][1] == 0
    assert snap["buckets"][8] == 0
    assert snap["buckets"][16] == 0


def test_histogram_record_boundaries():
    h = Histogram()
    h.record(0.005)  # Boundary, should go in bucket 0
    h.record(1000)  # Boundary, should go in bucket 16
    snap = h.snapshot()
    assert snap["buckets"][0] == 1
    assert snap["buckets"][16] == 1
    assert snap["count"] == 2


def test_histogram_record_inf():
    h = Histogram()
    h.record(float("inf"))  # Should go in the last bucket
    snap = h.snapshot()
    assert snap["buckets"][-1] == 1
    assert snap["max"] == float("inf")
    assert snap["count"] == 1


def test_histogram_snapshot_consistency():
    h = Histogram()
    h.record(10)
    snap1 = h.snapshot()
    snap2 = h.snapshot()  # Should be identical if no records between
    assert snap1 == snap2
    h.record(20)
    snap3 = h.snapshot()
    assert snap1 != snap3
    assert snap3["count"] == 2


# Test MetricsRegistry
def test_registry_get_key():
    registry = MetricsRegistry()
    assert registry._get_key("my_metric", None) == "my_metric"
    assert registry._get_key("my_metric", {"tag1": "val1"}) == "my_metric[tag1=val1]"
    assert (
        registry._get_key("my_metric", {"tag2": "val2", "tag1": "val1"})
        == "my_metric[tag1=val1,tag2=val2]"
    )  # Sorted
    assert (
        registry._get_key("my_metric", {"a": "1", "c": "3", "b": "2"})
        == "my_metric[a=1,b=2,c=3]"
    )


def test_registry_counter():
    registry = MetricsRegistry()
    c1 = registry.counter("reqs")
    c2 = registry.counter("reqs")
    assert c1 is c2  # Should return the same instance
    c1.inc()
    assert c2.get() == 1


def test_registry_gauge():
    registry = MetricsRegistry()
    g1 = registry.gauge("temp")
    g2 = registry.gauge("temp")
    assert g1 is g2
    g1.set(37.5)
    assert g2.get() == 37.5


def test_registry_histogram():
    registry = MetricsRegistry()
    h1 = registry.histogram("latency")
    h2 = registry.histogram("latency")
    assert h1 is h2
    h1.record(50)
    snap = h2.snapshot()
    assert snap["count"] == 1
    assert snap["sum"] == 50


def test_registry_tagged_metrics():
    registry = MetricsRegistry()
    c_a = registry.counter("hits", {"target": "a"})
    c_b = registry.counter("hits", {"target": "b"})
    c_a_again = registry.counter("hits", {"target": "a"})

    assert c_a is not c_b
    assert c_a is c_a_again
    c_a.inc(1)
    c_b.inc(5)

    assert c_a.get() == 1
    assert c_b.get() == 5


def test_registry_different_types_same_name():
    registry = MetricsRegistry()
    c = registry.counter("value")
    with pytest.raises(
        TypeError, match="Metric 'value' exists but has type Counter, requested Gauge"
    ):
        # Attempting to get a gauge with the same name should raise TypeError
        g = registry.gauge("value")


def test_registry_snapshot():
    registry = MetricsRegistry()
    registry.counter("c1").inc(10)
    registry.gauge("g1", {"t": "v"}).set(99.9)
    registry.histogram("h1").record(123)
    registry.histogram("h1").record(456)

    snap = registry.get_snapshot()

    assert "c1" in snap
    assert snap["c1"] == 10

    assert "g1[t=v]" in snap
    assert snap["g1[t=v]"] == 99.9

    assert "h1" in snap
    hist_snap = snap["h1"]
    assert isinstance(hist_snap, dict)
    assert hist_snap["count"] == 2
    assert hist_snap["sum"] == 123 + 456


# Test Metrics Facade (basic singleton access)
def test_metrics_singleton():
    m1 = Metrics.get()
    m2 = Metrics.get()
    assert m1 is m2


def test_metrics_facade_methods():
    # Reset singleton for test isolation if possible (tricky without direct access/reset method)
    # Instead, use unique names to avoid cross-test contamination
    metrics = Metrics.get()
    registry = metrics._registry  # Access internal registry for verification

    metrics.counter("facade_test_counter", 3, {"a": "1"})
    metrics.gauge("facade_test_gauge", 1.23, {"b": "2"})
    metrics.histogram("facade_test_hist", 45.6, {"c": "3"})

    snap = registry.get_snapshot()

    assert "facade_test_counter[a=1]" in snap
    assert snap["facade_test_counter[a=1]"] == 3

    assert "facade_test_gauge[b=2]" in snap
    assert snap["facade_test_gauge[b=2]"] == 1.23

    assert "facade_test_hist[c=3]" in snap
    hist_snap = snap["facade_test_hist[c=3]"]
    assert hist_snap["count"] == 1
    assert hist_snap["sum"] == 45.6


def test_metrics_timed():
    metrics = Metrics.get()
    registry = metrics._registry
    metric_name = "facade_timed_test"
    hist_name = metric_name + "_duration_ms"
    tags = {"timed": "yes"}

    with metrics.timed(metric_name, tags):
        time.sleep(0.01)  # Sleep for ~10ms

    snap = registry.get_snapshot()

    assert hist_name + "[timed=yes]" in snap
    hist_snap = snap[hist_name + "[timed=yes]"]
    assert hist_snap["count"] == 1
    assert hist_snap["sum"] > 5  # Should be >= 10ms, allow for scheduler variance
    assert hist_snap["sum"] < 50  # Sanity check upper bound
