"""Real Snakemake 9 scheduling, explicitly fake manager (no native Slurm claim)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def test_real_snakemake_releases_one_donor_before_other_source_exists(tmp_path):
    plugin_root = Path(__file__).parents[1]
    stub = tmp_path/'stub'; stub.mkdir()
    (stub/'zslurm_shared.py').write_text('''
def resolve_instance_name(instance): return instance
def get_config(**kwargs): return {}
def get_job_url(**kwargs): return 'http://no-manager.invalid/RPC2'
class TimeoutServerProxy:
    def __init__(self,*args,**kwargs): pass
    def __getattr__(self,name):
        raise AssertionError('External receipts must not query/submit/cancel real jobs: '+name)
''')
    sources={name:tmp_path/(name+'.source.json') for name in ('A','B')}
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='ZSLURM_EXTERNAL_RECEIPTS_V1',poll_seconds=30,
        targets={name:dict(source=str(source),destination=str(tmp_path/(name+'.ready.json')),
                           expected=dict(sample=name,complete=True)) for name,source in sources.items()})))
    sources['A'].write_text(json.dumps(dict(sample='A',complete=True)))
    snake=tmp_path/'Snakefile'
    snake.write_text('''
from pathlib import Path
localrules: all, consume
rule all:
    input: ['A.done','B.done']
rule await_external:
    input: str(Path('manifest.json').resolve())
    output: str(Path('.').resolve())+'/{sample}.ready.json'
    resources:
        zslurm_external_receipts=str(Path('manifest.json').resolve()),
        zslurm_external_key=lambda wc: wc.sample
    shell: 'exit 97'
rule consume:
    input: str(Path('.').resolve())+'/{sample}.ready.json'
    output: '{sample}.done'
    shell: 'touch {output:q}'
''')
    stopped=threading.Event(); released=[]
    def publish_second():
        deadline=time.monotonic()+65
        while not stopped.wait(.1) and time.monotonic()<deadline:
            if (tmp_path/'A.done').exists():
                assert not sources['B'].exists() and not (tmp_path/'B.done').exists()
                released.append('A completed while B missing')
                sources['B'].write_text(json.dumps(dict(sample='B',complete=True)))
                return
    thread=threading.Thread(target=publish_second,daemon=True);thread.start()
    env=dict(os.environ,PYTHONPATH=str(stub)+os.pathsep+str(plugin_root))
    try:
        result=subprocess.run([sys.executable,'-m','snakemake','--snakefile',str(snake),
            '--executor','zslurm','--zslurm-instance','isolated-test','--jobs','4','--cores','1',
            '--seconds-between-status-checks','1','--latency-wait','1'],
            cwd=tmp_path,env=env,text=True,capture_output=True,timeout=85)
    finally:
        stopped.set();thread.join(timeout=2)
    assert result.returncode==0,result.stdout+result.stderr
    assert released==['A completed while B missing']
    assert (tmp_path/'B.done').exists()
    for name in sources:
        witness=json.loads((tmp_path/(name+'.ready.json')).read_text())
        assert witness['worker_allocated'] is False
