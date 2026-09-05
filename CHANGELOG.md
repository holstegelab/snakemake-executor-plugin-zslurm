## Unreleased

### Added

* Add pipeline-wide --zslurm-priority and propagate it to the zslurm manager.
* Forward the Snakemake dcache_transfer_slots resource to the zslurm manager.
* Keep default-priority submissions compatible with older zslurm managers.

### Changed

* Submit S3 downloads only to the independent S3 transfer pool instead of also
  consuming dCache fallback slots.
* Accept Snakemake's portable `runtime` (minutes) and `cpus_per_task` resource
  names while retaining `time` (seconds) and `n` as explicit ZSlurm overrides.
* Export the ZSlurm instance, workflow owner, and runtime manager endpoint to
  logical jobs so provenance can authenticate their unique job IDs.

## 1.0.0 (2024-11-13)


### Miscellaneous Chores

* release 1.0.0
