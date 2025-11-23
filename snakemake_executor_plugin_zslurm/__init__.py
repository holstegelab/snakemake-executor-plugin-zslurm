__author__ = "Marc Hulsman"
__copyright__ = "Copyright 2024, Marc Hulsman"
__email__ = "m.hulsman1@amsterdamumc.nl"
__license__ = "MIT"

from dataclasses import dataclass, field
import os
import shlex
import subprocess
import sys
import threading
import time
import socket
import http.client as httplib
import time
import asyncio
import uuid
from typing import Generator, List, Optional

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.utils import async_lock
from snakemake_interface_executor_plugins.settings import (
    ExecutorSettingsBase,
    CommonSettings,
)
from snakemake_interface_executor_plugins.jobs import (
    JobExecutorInterface,
)
from snakemake.common.tbdstring import TBDString
from os import fspath
from collections.abc import Mapping, Sequence

# Import the zslurm_shared module, should be already installed
try:
    import zslurm_shared
except ImportError:
    raise ImportError(
        "The zslurm executor requires the zslurm_shared module to be installed. "
        "Please install it by installing the zslurm package."
    )


def ensure_int(value, msg=None):
    try:
        return int(value)
    except ValueError:
        raise ValueError(msg or f"Expected an integer, but got {value}")


def ensure_float(value, msg=None):
    try:
        return float(value)
    except ValueError:
        raise ValueError(msg or f"Expected a float, but got {value}")


# Optional:
# define additional settings for your executor
# They will occur in the Snakemake CLI as --<executor-name>-<param-name>
# Omit this class if you don't need any.
@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the zslurm configuration file. Otherwise looks in order for local zslurm.config file, then ~/.zslurm file.",
            "required": False,
        },
    )
    instance: Optional[str] = field(
        default=None,
        metadata={"help": "Instance of the zslurm server. Default is None."},
    )


# Required:
# Specify common settings shared by various executors.
common_settings = CommonSettings(
    non_local_exec=True,
    implies_no_shared_fs=False,
    job_deploy_sources=False,
    pass_default_storage_provider_args=True,
    pass_default_resources_args=True,
    pass_envvar_declarations_to_cmd=True,
    auto_deploy_default_storage_provider=False,
)

def to_primitive(x):
    if x is None:
        return x
    # Handle Snakemake's TBDString first (it subclasses str and is not XML-RPC serializable)
    if isinstance(x, TBDString) or type(x).__name__ == "TBDString":
        return str(x)
    if isinstance(x, (str, int, float, bool)):
        return x
    if hasattr(x, "__fspath__"):
        return fspath(x)
    if isinstance(x, Mapping):
        return {to_primitive(k): to_primitive(v) for k, v in x.items()}
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return [to_primitive(i) for i in x]
    return str(x)


# Required:
# Implementation of your executor
class Executor(RemoteExecutor):
    def __post_init__(self):
        #set _owner_id to a random uuid
        self._owner_id = str(uuid.uuid4())
        cfg_path = self.workflow.executor_settings.config_file
        instance = getattr(self.workflow.executor_settings, "instance", None)

        self.zslurm_config = zslurm_shared.get_config(
            config_path=cfg_path,
            instance=instance,  # optional; safe to keep
        )

        job_url = zslurm_shared.get_job_url(
            instance=instance  # config no longer needed here
        )
        self.zslurm_server = zslurm_shared.TimeoutServerProxy(job_url, allow_none=True)

    def run_job(self, job: JobExecutorInterface):
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.

        try:
            wildcard_str = "_".join(job.wildcards) if job.wildcards else ""
        except AttributeError:
            wildcard_str = ""

        # generic part of a submission string

        # we use a run_uuid as the job-name, to allow `--name`-based
        # filtering in the job status checks (`sacct --name` and `squeue --name`)
        comment_str = f"{wildcard_str} jobid:{job.jobid}"
        job_name = job.name
        partition = job.resources.get("partition", "compute")
        cmd = self.format_job_exec(job)

        reqtime = ensure_float(
            job.resources.get("time", 3600), f"time must be an number for {job_name}"
        )
        cores = ensure_float(
            job.resources.get("n", 1), f"n must be an number for {job_name}"
        )
        mem = ensure_float(
            job.resources.get("mem_mb", 500), f"mem_mb must be an number for {job_name}"
        )

        arch_use_add = ensure_float(
            job.resources.get("arch_use_add", 0),
            f"arch_use_add must be an number for {job_name}",
        )
        arch_use_remove = ensure_float(
            job.resources.get("arch_use_remove", 0),
            f"arch_use_remove must be an number for {job_name}",
        )
        dcache_use_add = ensure_float(
            job.resources.get("dcache_use_add", 0),
            f"dcache_use_add must be an number for {job_name}",
        )
        dcache_use_remove = ensure_float(
            job.resources.get("dcache_use_remove", 0),
            f"dcache_use_remove must be an number for {job_name}",
        )
        active_use_add = ensure_float(
            job.resources.get("active_use_add", 0),
            f"active_use_add must be an number for {job_name}",
        )
        active_use_remove = ensure_float(
            job.resources.get("active_use_remove", 0),
            f"active_use_remove must be an number for {job_name}",
        )

        cwd = self.workflow.workdir_init

        env = dict(os.environ)

        # Remove SNAKEMAKE_PROFILE from environment as the snakemake call inside
        # of the cluster job must run locally (or complains about missing -j).
        env.pop("SNAKEMAKE_PROFILE", None)

        limit_auto_threads = job.resources.get("limit_auto_threads", 0)
        try:
            limit_auto_threads = int(limit_auto_threads)
        except ValueError:
            raise WorkflowError("The limit_auto_threads resource must be an integer.")

        if limit_auto_threads >= 1:
            t = str(limit_auto_threads)
            env["OMP_NUM_THREADS"] = t
            env["OPENBLAS_NUM_THREADS"] = t
            env["MKL_NUM_THREADS"] = t
            env["VECLIB_MAXIMUM_THREADS"] = t
            env["NUMEXPR_NUM_THREADS"] = t

        info_input_mb = job.resources.get("input_mb", 0)
        info_output_file = str(job.output[0]) if job.output else None
        dependency = None
        # deps = " ".join(
        #    self.external_jobid[f] for f in job.input if f in self.external_jobid
        # )

        # SSD resources
        ssd_use = str(job.resources.get("ssd_use", "no"))
        ssd_gb = ensure_float(
            job.resources.get("ssd_gb", 0), f"ssd_gb must be an float for {job_name}"
        )

        requeue = 0  # done by snakemake
        attempt = 4
        while attempt > 0:
            try:
                s = self.zslurm_server

                slurm_jobid = s.submit_job(
                    to_primitive(job_name),
                    to_primitive(cmd),
                    to_primitive(cwd),
                    to_primitive(env),
                    to_primitive(cores),
                    to_primitive(mem),
                    to_primitive(reqtime),
                    to_primitive(requeue),
                    to_primitive(dependency),
                    to_primitive(arch_use_add),
                    to_primitive(arch_use_remove),
                    to_primitive(dcache_use_add),
                    to_primitive(dcache_use_remove),
                    to_primitive(active_use_add),
                    to_primitive(active_use_remove),
                    to_primitive(partition),
                    to_primitive(info_input_mb),
                    to_primitive(info_output_file),
                    to_primitive(comment_str),
                    to_primitive(ssd_use),
                    to_primitive(ssd_gb),
                    to_primitive(self._owner_id),
                )
                break
            except (socket.error, httplib.HTTPException, AttributeError) as serror:
                attempt -= 1
                if attempt == 0:
                    raise WorkflowError(
                        f"ZSLURM submission failed. The error message was {serror}"
                    )

                time.sleep(15)

        self.logger.info(
            f"Job {job.jobid}-{job_name} has been submitted with ZSlurm jobid {slurm_jobid} "
        )
        self.report_job_submission(SubmittedJobInfo(job, external_jobid=slurm_jobid))

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> Generator[SubmittedJobInfo, None, None]:
        # check if server has already initialized
        if not hasattr(self, "zslurm_server"):
            for active_job in active_jobs:
                yield active_job

        try:
            s = self.zslurm_server
            last_done = getattr(self, "_last_seen_done_jobid", None)

            if last_done is None:
                zslurm_done_jobs = s.list_done_jobs(None, self._owner_id)
            else:
                zslurm_done_jobs = s.list_done_jobs(last_done, self._owner_id)

            zslurm_active_jobs = s.list_jobs(self._owner_id)
        except (
            socket.error,
            socket.timeout,
            httplib.HTTPException,
            AttributeError,
            TimeoutError,
            asyncio.TimeoutError,
        ) as serror:
            self.logger.warning(
                f"ZSLURM job status check failed. The error message was {serror}"
            )
            if (
                self.wait
            ):  # if wait is False, we are exiting, and zslurm has likely been shut down.
                for active_job in active_jobs:  # assume all are still running
                    yield active_job
            return

        if zslurm_done_jobs:
            try:
                self._last_seen_done_jobid = zslurm_done_jobs[-1][0]
            except (IndexError, TypeError, KeyError):
                pass

        done_job_ids_state = {d[0]: d[2] for d in zslurm_done_jobs}
        active_job_ids_state = {a[0]: a[2] for a in zslurm_active_jobs}

        missing_status = []
        failed_states = {"CANCELLED", "FAILED", "TIMEOUT", "ERROR"}
        running_states = {"RUNNING", "PENDING", "ASSIGNED", "REQUEUED"}
        done_states = {"FINISHED", "COMPLETED"}

        any_finished = False

        for active_job in active_jobs:
            done = False
            if active_job.external_jobid in done_job_ids_state:
                state = done_job_ids_state[active_job.external_jobid]
                done = True
                any_finished = True
            elif active_job.external_jobid in active_job_ids_state:
                state = active_job_ids_state[active_job.external_jobid]
            else:
                # job not found in zslurm??
                missing_status.append(active_job.external_jobid)
                yield active_job  # for now
                continue

            if state in done_states:
                self.report_job_success(active_job)
            elif state in failed_states:
                msg = f"Job {active_job.external_jobid} failed with state {state}"
                self.report_job_error(active_job, msg=msg)
            elif state in running_states:
                # still active, yield again
                yield active_job
            else:
                if done:
                    self.report_job_error(
                        active_job,
                        msg=f"Job {active_job.external_jobid} has ended with an unknown state {state}. Assuming failure. Update the ZSlurm executor.",
                    )
                else:
                    self.logger.warning(
                        f"Job {active_job.external_jobid} has an unknown state {state}. Update the zslurm executor."
                    )
                    yield active_job

        if missing_status:
            self.logger.warning(
                f"Unable to get the status of all active jobs that should be "
                f"in zslurm.\n"
                f"The jobs with the following slurm job ids were previously seen "
                "but zslurm doesn't report them any more:\n"
                f"{missing_status}\n"
            )

        if not any_finished:
            self.next_seconds_between_status_checks = min(
                self.next_seconds_between_status_checks + 10, 180
            )
        else:
            self.next_seconds_between_status_checks = 30

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        self.logger.info(
            "ZSlurm executor does not implement job cancelling. Will exit after "
            "finishing currently running jobs."
        )
        self.shutdown()


    #overload to prevent snakemake from exiting on cancelling before all jobs are finished. 
    async def _wait_for_jobs(self):
        await asyncio.sleep(
            self.workflow.executor_plugin.common_settings.init_seconds_before_status_checks
        )
        while True:
            async with async_lock(self.lock):
                active_jobs = list(self.active_jobs)
                self.active_jobs.clear()

            still_active_jobs = [
                job_info async for job_info in self.check_active_jobs(active_jobs)
            ]
            async with async_lock(self.lock):
                if not self.wait and not still_active_jobs:
                    return
                # re-add the remaining jobs to active_jobs
                self.active_jobs.extend(still_active_jobs)
            await self.sleep()
