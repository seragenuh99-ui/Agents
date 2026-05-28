"""Tests for run.py task catalog and continuous12 suite."""

from experiments.benchmark_suites import get_suite, resolve_tasks
from experiments.tasks import ALL_TASK_GROUPS


def test_continuous12_has_at_least_12_tasks():
    tasks = get_suite("continuous12")
    assert len(tasks) >= 12


def test_continuous12_tasks_are_sequential_chain():
    tasks = get_suite("continuous12")
    ids = [t["task_id"] for t in tasks]
    assert ids[0] == "cont_task_1"
    assert ids[-1] == "cont_task_12"
    last = tasks[-1]["description"].lower()
    assert "prior" in last or "previous" in last or "1-11" in last


def test_resolve_tasks_by_ids():
    tasks = resolve_tasks(task_ids=["e1_solar_basics", "e4_renewable_comparison"])
    assert len(tasks) == 2


def test_all_task_groups_continuous12():
    assert len(ALL_TASK_GROUPS["continuous12"]) == 12
