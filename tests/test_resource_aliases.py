import pytest

from snakemake_executor_plugin_zslurm import (
    resolve_cores,
    resolve_runtime_seconds,
)


def test_native_time_is_seconds_and_has_precedence():
    assert resolve_runtime_seconds({"time": 90, "runtime": 12}) == 90.0


def test_standard_runtime_minutes_are_converted_to_seconds():
    assert resolve_runtime_seconds({"runtime": 12}) == 720.0


def test_runtime_default_is_one_hour():
    assert resolve_runtime_seconds({}) == 3600.0


@pytest.mark.parametrize("value", [0, -1, "broken"])
def test_invalid_runtime_is_rejected(value):
    with pytest.raises(ValueError):
        resolve_runtime_seconds({"runtime": value})


def test_native_core_count_has_precedence():
    resources = {"n": 1.5, "cpus_per_task": 4, "_cores": 8}
    assert resolve_cores(resources, threads=6) == 1.5


def test_cpus_per_task_preserves_site_memory_core_policy():
    resources = {"cpus_per_task": 4, "_cores": 2}
    assert resolve_cores(resources, threads=2) == 4.0


def test_rule_threads_are_portable_fallback():
    assert resolve_cores({"_cores": 1}, threads=3) == 3.0


def test_internal_cores_and_default_are_supported():
    assert resolve_cores({"_cores": 2}) == 2.0
    assert resolve_cores({}) == 1.0


@pytest.mark.parametrize("value", [0, -1, "broken"])
def test_invalid_core_count_is_rejected(value):
    with pytest.raises(ValueError):
        resolve_cores({"cpus_per_task": value})
