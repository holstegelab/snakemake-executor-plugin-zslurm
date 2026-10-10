"""Submission transport failures must not duplicate logical work."""
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

import pytest

import snakemake_executor_plugin_zslurm as plugin
from snakemake_interface_common.exceptions import WorkflowError


def executor(server, *, priority=100, cluster='spider'):
    result = object.__new__(plugin.Executor)
    result.workflow = SimpleNamespace(workdir_init='/work')
    result.format_job_exec = lambda job: 'python -m workflow'
    result.envvars = lambda: {'SLURM_CLUSTER_NAME': cluster}
    result._owner_id = 'workflow-owner'
    result._zslurm_instance = 'logical-name'
    result._zslurm_job_url = 'http://old:12345/RPC2'
    result._zslurm_priority = priority
    result.zslurm_server = server
    result.logger = Mock()
    result.report_job_submission = Mock()
    return result


def job():
    return SimpleNamespace(name='call', jobid=17, wildcards=[], threads=2,
                           resources={'mem_mb': 16000, 'runtime': 360}, output=['result'])


class Ledger:
    def __init__(self, *, lose_first=False):
        self.calls = []
        self.accepted = {}
        self.lose_first = lose_first

    def submit_job(self, *args):
        self.calls.append(args)
        # Legacy tokenless requests really would be accepted separately. Keep
        # this path so the socket regression reproduces duplication on the
        # pre-fix executor, rather than just rejecting its old call signature.
        token = args[22] if len(args) > 22 else 'legacy-' + str(len(self.calls))
        if token not in self.accepted:
            self.accepted[token] = str(len(self.accepted) + 1)
        if self.lose_first:
            self.lose_first = False
            raise ConnectionResetError('accepted but response lost')
        return self.accepted[token]


@pytest.fixture(autouse=True)
def no_retry_delays(monkeypatch):
    monkeypatch.setattr(plugin.time, 'sleep', Mock())
    monkeypatch.setattr(plugin.zslurm_shared, 'get_job_url',
                        lambda **kwargs: 'http://old:12345/RPC2')


@pytest.mark.parametrize('cluster', ['spider', 'snellius'])
@pytest.mark.parametrize('priority', [100, 0, 200])
def test_lost_reply_reuses_token_and_preserves_submission(cluster, priority):
    ledger = Ledger(lose_first=True)
    run = executor(ledger, priority=priority, cluster=cluster)
    run.run_job(job())
    assert len(ledger.accepted) == 1
    assert len(ledger.calls) == 2
    assert ledger.calls[0] == ledger.calls[1]
    args = ledger.calls[0]
    assert args[0:3] == ('call', 'python -m workflow', '/work')
    assert args[4:7] == (2.0, 16000.0, 21600.0)
    assert args[21] == 'workflow-owner'
    assert args[22].startswith('snakemake:workflow-owner:')
    assert args[3]['SLURM_CLUSTER_NAME'] == cluster
    assert args[3]['ZSLURM_OWNER_ID'] == 'workflow-owner'
    assert len(args) == (23 if priority == 100 else 24)
    if priority != 100:
        assert args[23] == priority
    assert run.report_job_submission.call_count == 1
    assert run.report_job_submission.call_args.args[0].external_jobid == '1'


@pytest.mark.parametrize('accepted_before_failure', [False, True])
def test_retry_rediscovers_handover_endpoint_without_new_token(monkeypatch, accepted_before_failure):
    # The ledger represents the manager's token table retained across handover.
    ledger = Ledger(lose_first=accepted_before_failure)
    old = Mock()
    old.submit_job.side_effect = (ledger.submit_job if accepted_before_failure
                                 else ConnectionRefusedError('old endpoint closed'))
    run = executor(old)
    monkeypatch.setattr(plugin.zslurm_shared, 'get_job_url',
                        lambda **kwargs: 'http://new:23456/RPC2')
    create_proxy = Mock(return_value=ledger)
    monkeypatch.setattr(plugin.zslurm_shared, 'TimeoutServerProxy', create_proxy)
    run.run_job(job())
    assert len(ledger.accepted) == 1
    assert old.submit_job.call_args.args == ledger.calls[-1]
    assert run._zslurm_job_url == 'http://new:23456/RPC2'
    assert run._owner_id == 'workflow-owner'
    create_proxy.assert_called_once_with('http://new:23456/RPC2', allow_none=True)
    assert run.report_job_submission.call_args.args[0].external_jobid == '1'


def test_discovery_failure_does_not_discard_accepted_request(monkeypatch):
    ledger = Ledger(lose_first=True)
    run = executor(ledger)
    monkeypatch.setattr(plugin.zslurm_shared, 'get_job_url', Mock(side_effect=OSError('not mounted')))
    run.run_job(job())
    assert len(ledger.accepted) == 1
    assert ledger.calls[0] == ledger.calls[1]


def test_exhausted_transport_attempts_report_uncertain_not_absent():
    server = Mock()
    server.submit_job.side_effect = TimeoutError('reply unavailable')
    run = executor(server)
    with pytest.raises(WorkflowError, match='may already exist.*reconcile') as error:
        run.run_job(job())
    calls = server.submit_job.call_args_list
    assert len(calls) == 4
    assert len({c.args[22] for c in calls}) == 1
    assert calls[0].args[22] in str(error.value)
    run.report_job_submission.assert_not_called()
    assert plugin.time.sleep.call_count == 3


def test_genuine_workflow_retry_gets_a_new_token():
    ledger = Ledger()
    run = executor(ledger)
    original = job()
    run.run_job(original)
    run.run_job(original)
    assert len(ledger.accepted) == 2
    assert ledger.calls[0][22] != ledger.calls[1][22]


def test_unsupported_manager_fails_closed_without_legacy_resubmission():
    server = Mock()
    server.submit_job.side_effect = xmlrpc.client.Fault(1, 'submit_job takes fewer arguments')
    run = executor(server)
    with pytest.raises(WorkflowError, match='must support idempotent'):
        run.run_job(job())
    server.submit_job.assert_called_once()
    run.report_job_submission.assert_not_called()
    plugin.time.sleep.assert_not_called()


def test_programming_error_is_not_retried_as_transport_failure():
    server = Mock()
    server.submit_job.side_effect = AttributeError('invalid client state')
    run = executor(server)
    with pytest.raises(AttributeError, match='invalid client state'):
        run.run_job(job())
    server.submit_job.assert_called_once()
    plugin.time.sleep.assert_not_called()


def test_real_xmlrpc_lost_reply_does_not_duplicate_accepted_job():
    class LoseOneReply(SimpleXMLRPCRequestHandler):
        def do_POST(self):
            if self.server.lose_reply:
                self.server.lose_reply = False
                data = self.rfile.read(int(self.headers['content-length']))
                self.server._marshaled_dispatch(data)
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            super().do_POST()

    ledger = Ledger()
    with SimpleXMLRPCServer(('127.0.0.1', 0), allow_none=True, logRequests=False,
                            requestHandler=LoseOneReply) as server:
        server.lose_reply = True
        server.register_function(ledger.submit_job, 'submit_job')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_address[1]}/RPC2'
            with xmlrpc.client.ServerProxy(url, allow_none=True) as proxy:
                run = executor(proxy)
                run._zslurm_job_url = url
                run.run_job(job())
            assert len(ledger.calls) == 2
            assert len(ledger.accepted) == 1
            assert ledger.calls[0] == ledger.calls[1]
            assert run.report_job_submission.call_args.args[0].external_jobid == '1'
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
