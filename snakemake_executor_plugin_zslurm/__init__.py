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
    ip: Optional[str] = field(
        default=None,
        metadata={"help": "IP address of the zslurm server. Default is localhost."},
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


# Required:
# Implementation of your executor
class Executor(RemoteExecutor):
    def __post_init__(self):
        config_file = self.workflow.executor_settings.config_file
        self.zslurm_config = zslurm_shared.get_config(config_file)
        if self.workflow.executor_settings.ip:
            ip = self.workflow.executor_settings.ip
        else:
            ip = "localhost"

        job_url = zslurm_shared.get_job_url(ip, self.zslurm_config)
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

        requeue = 0  # done by snakemake
        attempt = 4
        while attempt > 0:
            try:
                s = self.zslurm_server
                slurm_jobid = s.submit_job(
                    job_name,
                    cmd,
                    cwd,
                    env,
                    cores,
                    mem,
                    reqtime,
                    requeue,
                    dependency,
                    arch_use_add,
                    arch_use_remove,
                    dcache_use_add,
                    dcache_use_remove,
                    active_use_add,
                    active_use_remove,
                    partition,
                    info_input_mb,
                    info_output_file,
                    comment_str,
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
            zslurm_done_jobs = s.list_done_jobs()
            zslurm_active_jobs = s.list_jobs()
        except (socket.error, httplib.HTTPException, AttributeError) as serror:
            self.logger.error(
                f"ZSLURM job status check failed. The error message was {serror}"
            )
            if (
                self.wait
            ):  # if wait is False, we are exiting, and zslurm has likely been shut down.
                for active_job in active_jobs:  # assume all are still running
                    yield active_job
            return

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
