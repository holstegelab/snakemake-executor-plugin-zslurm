import asyncio
import xmlrpc.client
from types import SimpleNamespace

from snakemake_executor_plugin_zslurm import (
    Executor,
    is_missing_rpc_method,
    parse_targeted_job_states,
)


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


class FakeExecutor:
    def __init__(self, server):
        self.zslurm_server = server
        self._owner_id = "owner-1"
        self.wait = True
        self.logger = RecordingLogger()
        self.next_seconds_between_status_checks = 30
        self.successes = []
        self.errors = []

    def report_job_success(self, job):
        self.successes.append(job.external_jobid)

    def report_job_error(self, job, msg=None):
        self.errors.append((job.external_jobid, msg))


def collect_status(executor, jobs):
    async def collect():
        return [job async for job in Executor.check_active_jobs(executor, jobs)]

    return asyncio.run(collect())


def test_targeted_status_recovers_terminal_job_outside_display_history():
    class Server:
        def get_job_states(self, jobids, owner):
            assert jobids == ["old", "live"]
            assert owner == "owner-1"
            return {
                "schema_version": 1,
                "states": {"old": "COMPLETED", "live": "RUNNING"},
                "terminal": ["old"],
                "unknown": [],
            }

        def list_done_jobs(self, *_args):
            raise AssertionError("legacy completion history must not be used")

        def list_jobs(self, *_args):
            raise AssertionError("legacy active-job listing must not be used")

    executor = FakeExecutor(Server())
    old = SimpleNamespace(external_jobid="old")
    live = SimpleNamespace(external_jobid="live")

    remaining = collect_status(executor, [old, live])

    assert executor.successes == ["old"]
    assert executor.errors == []
    assert remaining == [live]


def test_targeted_status_reports_terminal_failure():
    class Server:
        def get_job_states(self, jobids, owner):
            assert jobids == ["42"]
            assert owner == "owner-1"
            return {
                "schema_version": 1,
                "states": {"42": "FAILED"},
                "terminal": ["42"],
                "unknown": [],
            }

    executor = FakeExecutor(Server())
    failed = SimpleNamespace(external_jobid="42")

    remaining = collect_status(executor, [failed])

    assert executor.successes == []
    assert executor.errors == [("42", "Job 42 failed with state FAILED")]
    assert remaining == []


def test_old_manager_fallback_advances_cursor_to_newest_result():
    class Server:
        def get_job_states(self, *_args):
            raise xmlrpc.client.Fault(1, 'method "get_job_states" is not supported')

        def list_done_jobs(self, last_seen, owner):
            assert last_seen is None
            assert owner == "owner-1"
            return [
                ["12", "new", "COMPLETED"],
                ["11", "old", "COMPLETED"],
            ]

        def list_jobs(self, owner):
            assert owner == "owner-1"
            return [["13", "live", "RUNNING"]]

    executor = FakeExecutor(Server())
    completed = SimpleNamespace(external_jobid="12")
    running = SimpleNamespace(external_jobid="13")

    remaining = collect_status(executor, [completed, running])

    assert executor.successes == ["12"]
    assert remaining == [running]
    assert executor._last_seen_done_jobid == "12"
    assert executor._targeted_status_retry_after > 0


def test_targeted_response_validation_and_missing_method_detection():
    states, terminal = parse_targeted_job_states(
        {
            "states": {7: "FAILED"},
            "terminal": [7],
        }
    )
    assert states == {"7": "FAILED"}
    assert terminal == {"7"}
    assert is_missing_rpc_method(
        xmlrpc.client.Fault(1, "method get_job_states is not supported"),
        "get_job_states",
    )
