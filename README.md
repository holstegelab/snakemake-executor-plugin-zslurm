# snakemake-executor-plugin-zslurm

A Snakemake executor plugin directly interfacing with the zslurm wrapper for Slurm systems.

Zslurm: https://github.com/holstegelab/zslurm

## Portable resource names

The executor accepts both the original ZSlurm resource names and the standard
names used by Snakemake's Slurm executor:

| ZSlurm resource | Portable alias | Units passed to ZSlurm |
|---|---|---|
| `time` | `runtime` | seconds (`runtime` is converted from minutes) |
| `n` | `cpus_per_task` | cores |

The native name takes precedence when both are set. If neither CPU resource is
present, rule `threads` and then Snakemake's internal `_cores` value are used.
This allows one workflow to keep executor-specific policy in separate profiles.

Every child also receives `ZSLURM_INSTANCE`, `ZSLURM_OWNER_ID`, and the
runtime-only `ZSLURM_JOB_URL`; the chief adds its unique `ZSLURM_JOB_ID`.
Downstream provenance code can therefore distinguish logical rules sharing one
outer `SLURM_JOB_ID` and authenticate their terminal state with the manager.

Connected Snakemake `group` jobs are submitted as one logical ZSlurm job. All
rules inside that group inherit the same `ZSLURM_JOB_ID`, as expected for one
scheduler allocation; rules outside the group receive their own logical ID.
The minimal `tests/fixtures/grouped_chain.smk` workflow can be used for an
end-to-end manager smoke test.

## Worker environment

Workflow `envvars:` declarations, `--envvars`, and storage-provider variables
are forwarded in the job's XML-RPC environment dictionary. They are not emitted
as shell `export ... &&` prefixes: ZSlurm chiefs launch the Python command
directly without a shell. Values containing spaces, quotes, or newlines are
preserved literally. Executor-owned identity, transfer-slot and thread-limit
variables are applied afterwards; `SNAKEMAKE_PROFILE` is removed for the worker.

`tests/fixtures/declared_env.smk` provides a small end-to-end environment smoke
test. It reads the supplied restart manifest and writes only `env-check.json`
in its test workdir. Run it in a separate scratch workdir, not a production run.


## Pipeline priority

Assign one integer priority to every job submitted by a Snakemake run:

    snakemake --executor zslurm --zslurm-priority 100 ...

In a Snakemake profile, use:

    executor: zslurm
    zslurm-priority: 100

Higher values are dispatched before lower values. The default is 100, so existing
profiles retain their current behaviour. The priority is attached to all jobs in
the run, including staging, download, upload, archive, and regular compute jobs.
The plugin forwards `dcache_download_slots` and `dcache_upload_slots` rule
resources, allowing the manager to enforce independent instance-wide transfer
limits while selecting the highest-priority waiting transfer. It also forwards
`s3_download_slots` exclusively to the separate S3 download pool, without
consuming a dCache slot. The plugin emits a conservative combined fallback for
directional dCache requests during a rolling upgrade; updated managers ignore
that fallback when directional metadata is present. The explicit legacy
`dcache_transfer_slots` resource is still
supported, but cannot be combined with a directional resource on the same job.
It does not preempt jobs that are
already running. Within an equal priority band, zslurm retains its existing
FIFO/LIFO and memory-packing behaviour.

Every submission requires a manager supporting the `idempotency_key` argument
of `submit_job`. A nondefault priority additionally requires its `priority`
argument. Unsupported managers receive a clear error; the executor does not
silently fall back to a submission that could be duplicated.

## Submission recovery

Transport retries reuse one unique submission token, including after endpoint
rediscovery for the same named instance. ZSlurm retains accepted tokens across
manager handover, so losing a reply does not create a second logical job.
Owner, priority, resources and command are retained. A genuine Snakemake job
retry uses a new token; this mechanism does not deduplicate new controller
processes or cold starts against earlier workflow attempts. Rediscovery assumes
the named instance still identifies the same manager lineage with its retained
token table; do not repoint it to an unrelated, empty replacement manager.

After four unconfirmed attempts the executor reports an uncertain outcome and
the token to reconcile; it does not claim the job was never accepted. Inspect
the manager before restarting that workflow. Running controllers retain their
already imported plugin until their next normal start; installing an update
does not hot-patch them.

## External receipt dependencies (shared filesystem)

An opt-in rule can wait for another workflow's small JSON receipt without
submitting a ZSlurm job or allocating a pilot worker. Declare string resources
`zslurm_external_receipts` (absolute manifest path) and `zslurm_external_key`
(entry key). The manifest must be a rule input; the rule's sole output is a
**distinct readiness witness**, never the external source. Its shell command
is not executed by this executor; use an explicit failure command so an
unsupported executor cannot silently claim success. Do not group observer
rules with compute rules or assign them `localrules`.

The manifest schema is `ZSLURM_EXTERNAL_RECEIPTS_V1`, with `poll_seconds`
(integer 30--3600) and `targets`, mapping each key to exactly `source`,
`destination` (absolute paths), and nonempty `expected` (JSON fields to match).
Nested object fields match recursively; lists and primitive types match exactly.
Use an expected schema, donor identity and source pins rather than just a
generic success flag. Files larger than 16 MiB are rejected. The witness schema
is `ZSLURM_EXTERNAL_READY_V1`; it records the manifest/source byte hashes and
explicitly grants no scientific validation. Consumers must authenticate their
actual source closure independently.

Missing, partially written or temporarily unreadable source receipts remain
pending. Changed manifests or valid receipts with wrong identity fields fail
the observing rule. Each donor can release independently; there is no implicit
population barrier. Shutdown drops only in-process observations: it does not
cancel or adopt the external producer. Original source files are never written
or removed. Existing witnesses are not overwritten. A failed producer with no
receipt therefore needs operator monitoring; it is never inferred as successful
or missing biological evidence. Ordinary job submission/status/drain behavior
is unchanged on Spider and Snellius.

Observers still occupy Snakemake logical `--jobs` slots. Set that limit above
the maximum simultaneous observers plus desired compute concurrency; the
observers themselves consume no ZSlurm worker slots. No manager upgrade is
required. `test_external_receipts_snakemake.py` runs a real Snakemake dependency
test with an intentionally fake, RPC-rejecting manager; it is not a native
cluster qualification.
