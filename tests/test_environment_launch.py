"""Exercise command formatting, RPC serialization and shell-free worker launch."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import xmlrpc.client

import pytest

from snakemake_executor_plugin_zslurm import Executor, common_settings


FIXTURE = Path(__file__).parent / "fixtures" / "declared_env.smk"


def capture_submission(workdir, declared):
    # Keep the real format_job_exec(), envvars(), and run_job(). Only replace
    # unrelated DAG arguments and the server: no manager is needed for this test.
    executor = object.__new__(Executor)
    executor.workflow = SimpleNamespace(
        executor_plugin=SimpleNamespace(common_settings=common_settings),
        workdir_init=str(workdir),
        spawned_job_args_factory=SimpleNamespace(
            envvars=lambda: declared,
            general_args=lambda **kwargs: "--nolock --force --force-use-threads",
            precommand=lambda **kwargs: "",
        ),
        group_settings=SimpleNamespace(local_groupid="local"),
    )
    executor.snakefile = str(FIXTURE)
    executor.get_job_args = lambda job: "--cores 1"
    executor.get_python_executable = lambda: shlex.quote(sys.executable)
    executor._owner_id = "test-owner"
    executor._zslurm_instance = "test-instance"
    executor._zslurm_job_url = "http://localhost:12345/test-only"
    executor._zslurm_priority = 100
    executor.logger = Mock()
    executor.zslurm_server = Mock()
    executor.zslurm_server.submit_job.return_value = "42"
    executor.report_job_submission = Mock()
    job = SimpleNamespace(
        name="env_probe",
        jobid=1,
        wildcards=[],
        threads=1,
        resources={"mem_mb": 256, "n": 0.1, "time": 120, "limit_auto_threads": 8},
        output=["env-check.json"],
    )
    executor.run_job(job)
    executor.zslurm_server.submit_job.assert_called_once()
    args = executor.zslurm_server.submit_job.call_args.args
    # Confirm the actual transport representation, including newlines/quotes.
    args, _ = xmlrpc.client.loads(xmlrpc.client.dumps(args, allow_none=True))
    return args[1], args[2], args[3]


@pytest.mark.parametrize("value", ["plain", "", "quotes '\" $literal `literal` ; &&\nsecond line"])
def test_declared_environment_reaches_shell_free_snakemake_worker(tmp_path, monkeypatch, value):
    manifest = tmp_path / "manifest with spaces and 'quotes'.json"
    manifest.write_text(json.dumps({"reused_samples": ["finished-sample"]}))
    declared = {
        "SHORT_READ_RESTART_MANIFEST": str(manifest),
        "ZSLURM_TEST_DECLARED": value,
        # Storage providers may supply variables not present in os.environ.
        "ZSLURM_TEST_PROVIDER": "provider-only value",
    }
    monkeypatch.setenv("SHORT_READ_RESTART_MANIFEST", str(manifest))
    monkeypatch.setenv("ZSLURM_TEST_DECLARED", value)
    monkeypatch.delenv("ZSLURM_TEST_PROVIDER", raising=False)
    monkeypatch.setenv("SNAKEMAKE_PROFILE", "must-not-reach-worker")
    monkeypatch.setenv("ZSLURM_INSTANCE", "stale-parent-instance")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")

    command, cwd, env = capture_submission(tmp_path, declared)
    argv = shlex.split(command)
    assert argv[0] == sys.executable  # Regression: previously this was 'export'.
    assert "&&" not in argv
    for key, expected in declared.items():
        assert env[key] == expected
    assert env["ZSLURM_INSTANCE"] == "test-instance"
    assert env["OMP_NUM_THREADS"] == "8"
    assert "SNAKEMAKE_PROFILE" not in env

    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    received = json.loads((tmp_path / "env-check.json").read_text())
    for key, expected in declared.items():
        assert received[key] == expected
    assert received["reused_samples"] == 1


def test_workflow_without_declarations_keeps_inherited_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ZSLURM_TEST_AMBIENT", "keep this")
    command, cwd, env = capture_submission(tmp_path, {})
    assert shlex.split(command)[0] == sys.executable
    assert env["ZSLURM_TEST_AMBIENT"] == "keep this"
    assert cwd == str(tmp_path)


def test_workflow_environment_overrides_stale_ambient_values(tmp_path, monkeypatch):
    monkeypatch.setenv("ZSLURM_TEST_PROVIDER", "stale ambient value")
    _, _, env = capture_submission(tmp_path, {"ZSLURM_TEST_PROVIDER": "resolved provider value"})
    assert env["ZSLURM_TEST_PROVIDER"] == "resolved provider value"
