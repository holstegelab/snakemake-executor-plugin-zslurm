import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import snakemake_executor_plugin_zslurm as plugin
from snakemake_executor_plugin_zslurm import external_receipts as external
from test_submission_recovery import executor


def fixture(tmp_path, name='S1'):
    source = tmp_path/'producer'/name/'lineage.json'
    output = tmp_path/'consumer'/name/'ready.json'
    manifest = tmp_path/(name+'.manifest.json')
    record = dict(source=str(source),destination=str(output),expected=dict(operation='LINEAGE',sample=name,phased=False))
    manifest.write_text(json.dumps(dict(schema=external.SCHEMA,poll_seconds=300,targets={name:record})))
    job = SimpleNamespace(name='await_lineage',jobid=3,input=[str(manifest)],output=[str(output)],
        resources=dict(zslurm_external_receipts=str(manifest),zslurm_external_key=name))
    return job, source, output, manifest


def publish(source, **overrides):
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text(json.dumps(dict(operation='LINEAGE',sample=source.parent.name,phased=False,**overrides)))


def check(run, jobs):
    async def consume():
        return [job async for job in run.check_active_jobs(jobs)]
    return asyncio.run(consume())


@pytest.mark.parametrize('cluster', ['spider','snellius'])
def test_external_donors_release_independently_without_workers_or_manager_queries(tmp_path, cluster):
    first, s1, o1, _ = fixture(tmp_path,'S1')
    second, _, o2, _ = fixture(tmp_path,'S2')
    server = Mock(); run = executor(server,cluster=cluster); run.wait = True
    run.report_job_success = Mock(); run.report_job_error = Mock()
    run.run_job(first); run.run_job(second)
    active = [call.args[0] for call in run.report_job_submission.call_args_list]
    assert check(run,active) == active
    publish(s1)
    for observer in run._external_receipts.values(): observer.next_check = 0
    assert check(run,active) == active[1:]
    assert o1.is_file() and not o2.exists()
    result = json.loads(o1.read_text())
    assert result['source']['sha256'] == hashlib.sha256(s1.read_bytes()).hexdigest()
    assert result['scientific_validation'] is False and result['worker_allocated'] is False
    assert result['external_producer_cancelled'] is False
    assert server.mock_calls == []
    run.report_job_success.assert_called_once_with(active[0])
    run.report_job_error.assert_not_called()


def test_bounded_poll_cadence_and_partial_json_never_mean_ready(tmp_path):
    job, source, out, _ = fixture(tmp_path)
    observer = external.from_job(job)
    source.parent.mkdir(parents=True);source.write_text('{"sample":')
    assert observer.check(now=1) is False
    publish(source)
    assert observer.check(now=300) is False
    assert not out.exists()
    assert observer.check(now=301) is True
    with pytest.raises(FileExistsError):observer.check(now=601)


@pytest.mark.parametrize('damage', ['sample','bool_type','manifest','destination','oversize'])
def test_external_wait_never_writes_source_or_accepts_changed_contract(tmp_path, damage):
    job, source, out, manifest = fixture(tmp_path)
    observer = external.from_job(job)
    publish(source)
    if damage == 'sample':
        body=json.loads(source.read_text());body['sample']='OTHER';source.write_text(json.dumps(body))
    elif damage == 'bool_type':
        body=json.loads(source.read_text());body['phased']=0;source.write_text(json.dumps(body))
    elif damage == 'manifest':manifest.write_text(manifest.read_text().replace('300','301'))
    elif damage == 'destination':out.parent.mkdir(parents=True);out.write_text('existing witness')
    elif damage == 'oversize':source.write_bytes(b' '*(external.MAX_BYTES+1))
    before=source.read_bytes()
    with pytest.raises((ValueError,FileExistsError)):observer.check(now=0)
    assert source.read_bytes()==before
    assert not out.exists() if damage!='destination' else out.read_text()=='existing witness'


def test_shutdown_releases_observers_without_waiting_for_external_producer(tmp_path):
    job, source, out, _ = fixture(tmp_path)
    server=Mock();run=executor(server);run.wait=True
    run.report_job_success=Mock();run.report_job_error=Mock();run.run_job(job)
    info=run.report_job_submission.call_args.args[0]
    run.wait=False
    assert check(run,[info])==[] and run._external_receipts=={}
    assert server.mock_calls==[] and not source.exists() and not out.exists()
    run.report_job_success.assert_not_called();run.report_job_error.assert_not_called()


def test_observer_ids_never_leak_into_real_job_rpc_and_real_jobs_keep_their_owner(tmp_path):
    job, _, _, _=fixture(tmp_path)
    server=Mock();run=executor(server);run.wait=True;run.next_seconds_between_status_checks=30
    run.run_job(job);info=run.report_job_submission.call_args.args[0]
    normal=SimpleNamespace(external_jobid='123')
    server.get_job_states.return_value=dict(schema_version=1,states={'123':'RUNNING'},terminal=[],unknown=[])
    assert check(run,[info,normal])==[info,normal]
    server.get_job_states.assert_called_once_with(['123'],'workflow-owner')
    server.submit_job.assert_not_called()


def test_transient_source_io_keeps_observation_pending(tmp_path,monkeypatch):
    job, source, out, _=fixture(tmp_path)
    observer=external.from_job(job);publish(source)
    original=Path.stat
    def fail(path,*args,**kwargs):
        if path==source:raise OSError('temporary network filesystem failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',fail)
    assert observer.check(now=0) is False and not out.exists()


@pytest.mark.parametrize('damage',['input','output','source_output','missing_key','poll','relative'])
def test_registration_requires_explicit_separate_witness_and_input(tmp_path,damage):
    job, source, out, manifest=fixture(tmp_path)
    body=json.loads(manifest.read_text())
    if damage=='input':job.input=[]
    elif damage=='output':job.output=[str(out),str(source)]
    elif damage=='source_output':body['targets']['S1']['destination']=str(source);job.output=[str(source)]
    elif damage=='missing_key':job.resources['zslurm_external_key']='OTHER'
    elif damage=='poll':body['poll_seconds']=1
    elif damage=='relative':body['targets']['S1']['source']='relative.json'
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError):external.from_job(job)
