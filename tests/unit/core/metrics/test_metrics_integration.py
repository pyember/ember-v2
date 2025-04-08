import time

import pytest

from ember.core.metrics.integration import ComponentMetrics, measure_time

# Assuming metrics facade and integration helpers are importable
from ember.core.metrics.metrics import Metrics


# Test @measure_time decorator
@pytest.fixture
def metrics_instance():
    # Provide a fresh Metrics instance for isolation if possible,
    # otherwise rely on unique metric names.
    # For simplicity, we use the global singleton here.
    return Metrics.get()


def test_measure_time_decorator(metrics_instance):
    registry = metrics_instance._registry
    metric_base_name = "decorated_func_test"
    hist_name = metric_base_name + "_duration_ms"
    tags = {"deco": "measure"}

    @measure_time(metric_base_name, tags=tags)
    def timed_function():
        time.sleep(0.015)  # ~15ms
        return "done"

    result = timed_function()
    assert result == "done"

    snap = registry.get_snapshot()
    key = registry._get_key(hist_name, tags)

    assert key in snap
    hist_snap = snap[key]
    assert hist_snap["count"] == 1
    assert hist_snap["sum"] > 10  # Should be >= 15ms
    assert hist_snap["sum"] < 60  # Sanity check


def test_measure_time_decorator_no_tags(metrics_instance):
    registry = metrics_instance._registry
    metric_base_name = "decorated_func_no_tags"
    hist_name = metric_base_name + "_duration_ms"

    @measure_time(metric_base_name)
    def timed_function_no_tags():
        time.sleep(0.011)  # ~11ms
        return "ok"

    result = timed_function_no_tags()
    assert result == "ok"

    snap = registry.get_snapshot()
    key = registry._get_key(hist_name, None)

    assert key in snap
    hist_snap = snap[key]
    assert hist_snap["count"] == 1
    assert hist_snap["sum"] > 6  # Should be >= 11ms
    assert hist_snap["sum"] < 55  # Sanity check


# Test ComponentMetrics
@pytest.fixture
def component_metrics(metrics_instance):
    # Create a ComponentMetrics instance for testing
    return ComponentMetrics(
        metrics_instance, "test_component", base_tags={"env": "test"}
    )


def test_component_metrics_init(component_metrics):
    assert component_metrics._component_name == "test_component"
    assert component_metrics._base_tags == {"env": "test"}


def test_component_metrics_count(component_metrics, metrics_instance):
    registry = metrics_instance._registry
    component_metrics.count("login_attempt", value=2, tags={"status": "fail"})

    snap = registry.get_snapshot()
    # Expected key: test_component.operations_total[env=test,operation=login_attempt,status=fail]
    expected_key = registry._get_key(
        "test_component.operations_total",
        {"env": "test", "operation": "login_attempt", "status": "fail"},
    )

    assert expected_key in snap
    assert snap[expected_key] == 2


def test_component_metrics_count_no_extra_tags(component_metrics, metrics_instance):
    registry = metrics_instance._registry
    component_metrics.count("page_view")

    snap = registry.get_snapshot()
    # Expected key: test_component.operations_total[env=test,operation=page_view]
    expected_key = registry._get_key(
        "test_component.operations_total", {"env": "test", "operation": "page_view"}
    )
    assert expected_key in snap
    assert snap[expected_key] == 1


def test_component_metrics_time(component_metrics, metrics_instance):
    registry = metrics_instance._registry
    component_metrics.time("db_query", 150.5, tags={"table": "users"})

    snap = registry.get_snapshot()
    # Expected key: test_component.duration_ms[env=test,operation=db_query,table=users]
    expected_key = registry._get_key(
        "test_component.duration_ms",
        {"env": "test", "operation": "db_query", "table": "users"},
    )

    assert expected_key in snap
    hist_snap = snap[expected_key]
    assert hist_snap["count"] == 1
    assert pytest.approx(hist_snap["sum"]) == 150.5


def test_component_metrics_gauge(component_metrics, metrics_instance):
    registry = metrics_instance._registry
    component_metrics.gauge("queue_depth", 42, tags={"queue": "priority"})

    snap = registry.get_snapshot()
    # Expected key: test_component.queue_depth[env=test,queue=priority]
    expected_key = registry._get_key(
        "test_component.queue_depth", {"env": "test", "queue": "priority"}
    )

    assert expected_key in snap
    assert snap[expected_key] == 42


def test_component_metrics_timed(component_metrics, metrics_instance):
    registry = metrics_instance._registry
    operation_name = "background_job"
    extra_tags = {"job_id": "123"}

    with component_metrics.timed(operation_name, tags=extra_tags):
        time.sleep(0.012)  # ~12ms

    snap = registry.get_snapshot()
    # Expected key: test_component.duration_ms[env=test,job_id=123,operation=background_job]
    expected_key = registry._get_key(
        "test_component.duration_ms",
        {"env": "test", "operation": operation_name, "job_id": "123"},
    )

    assert expected_key in snap
    hist_snap = snap[expected_key]
    assert hist_snap["count"] == 1
    assert hist_snap["sum"] > 7  # Should be >= 12ms
    assert hist_snap["sum"] < 58  # Sanity check
