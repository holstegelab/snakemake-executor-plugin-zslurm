"""Observe explicitly declared external JSON receipts without allocating workers.

The external workflow keeps ownership. We only publish a distinct readiness
witness; consumers remain responsible for scientific/source authentication.
Missing, partially written or temporarily inaccessible receipts remain pending.
No timeout means a failed producer can never be mistaken for absent evidence.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

SCHEMA = 'ZSLURM_EXTERNAL_RECEIPTS_V1'
WITNESS_SCHEMA = 'ZSLURM_EXTERNAL_READY_V1'
MAX_BYTES = 16*1024*1024


def identity(path, raw):
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('absolute non-traversing external receipt path required')
    return Path(value).resolve()


def bounded_read(path):
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError('external JSON receipt exceeds 16 MiB')
    return raw


def matches(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(actual[k], v) for k,v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(matches(a,b) for a,b in zip(actual,expected))
    return type(actual) is type(expected) and actual == expected


@dataclass
class ExternalReceipt:
    manifest: dict
    key: str
    source: Path
    destination: Path
    expected: dict
    poll_seconds: int
    next_check: float = 0.

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        if now < self.next_check:
            return False
        self.next_check = now+self.poll_seconds
        # A changed declaration is an error, not a new or more permissive wait.
        raw = bounded_read(Path(self.manifest['path']))
        if identity(Path(self.manifest['path']), raw) != self.manifest:
            raise ValueError('external receipt manifest changed while waiting')
        try:
            before = self.source.stat()
            raw = bounded_read(self.source)
            after = self.source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                return False
            body = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not matches(body, self.expected):
            raise ValueError('external receipt does not match its declared identity fields')
        witness = dict(schema=WITNESS_SCHEMA, manifest=self.manifest, key=self.key,
            source=identity(self.source,raw), scientific_validation=False,
            external_producer_cancelled=False, worker_allocated=False)
        self.destination.parent.mkdir(parents=True,exist_ok=True)
        # This path belongs solely to the observing workflow. Never replace an
        # existing witness or write/remove any file belonging to the producer.
        with self.destination.open('x') as stream:
            json.dump(witness,stream,sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        return True


def from_job(job):
    path = absolute(job.resources['zslurm_external_receipts'])
    key = job.resources.get('zslurm_external_key')
    if not isinstance(key,str) or not key:
        raise ValueError('explicit external receipt key required')
    raw = bounded_read(path); body = json.loads(raw)
    period = body.get('poll_seconds')
    if (body.get('schema') != SCHEMA or type(period) is not int or not 30 <= period <= 3600
            or not isinstance(body.get('targets'),dict) or key not in body['targets']):
        raise ValueError('explicit bounded-cadence external receipt manifest required')
    record = body['targets'][key]
    if (not isinstance(record,dict) or set(record) != {'source','destination','expected'}
            or not isinstance(record['expected'],dict) or not record['expected']):
        raise ValueError('external source, fresh destination and identity fields required')
    source, destination = absolute(record['source']), absolute(record['destination'])
    if (source == destination or destination == path or destination in source.parents
            or source in destination.parents or list(map(lambda p:str(absolute(str(p))),job.output)) != [str(destination)]
            or str(path) not in {str(absolute(str(p))) for p in job.input}):
        raise ValueError('external witness must be the only output, separate from its declared source and manifest')
    if destination.exists():
        raise ValueError('external readiness witness already exists; never overwrite')
    return ExternalReceipt(identity(path,raw),key,source,destination,record['expected'],period)
