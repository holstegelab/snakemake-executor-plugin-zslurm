from types import SimpleNamespace

from snakemake_executor_plugin_zslurm import Executor


def test_workflow_exception_is_reported_before_potentially_long_drain():
    events=[]
    executor=SimpleNamespace(logger=SimpleNamespace(
        error=lambda message:events.append(('error',message)),
        info=lambda message:events.append(('info',message))),
        shutdown=lambda:events.append(('drain',None)))
    try:
        raise ModuleNotFoundError('missing rule-only dependency')
    except ModuleNotFoundError:
        Executor.cancel_jobs(executor, [])
    assert [e[0] for e in events]==['error','info','drain']
    assert 'ModuleNotFoundError: missing rule-only dependency' in events[0][1]
    assert 'test_workflow_exception_is_reported' in events[0][1]


def test_normal_shutdown_still_drains_without_fabricating_an_error():
    events=[]
    executor=SimpleNamespace(logger=SimpleNamespace(
        error=lambda message:events.append('error'),info=lambda message:events.append('info')),
        shutdown=lambda:events.append('drain'))
    Executor.cancel_jobs(executor, [])
    assert events==['info','drain']
