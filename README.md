# snakemake-executor-plugin-zslurm

A Snakemake executor plugin directly interfacing with the zslurm wrapper for Slurm systems.

Zslurm: https://github.com/holstegelab/zslurm


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
limits while selecting the highest-priority waiting transfer. The legacy
`dcache_transfer_slots` resource is still forwarded, but cannot be combined
with a directional resource on the same job. It does not preempt jobs that are
already running. Within an equal priority band, zslurm retains its existing
FIFO/LIFO and memory-packing behaviour.

The default priority 100 remains compatible with older zslurm managers. Any different priority requires a manager version that supports the extended
submit_job API; the plugin
reports a clear error if the running manager has not yet been updated.
