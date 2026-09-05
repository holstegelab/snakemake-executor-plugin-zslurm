"""Minimal executor smoke test for a connected Snakemake group job."""


rule all:
    input:
        "group.done"


rule grouped_first:
    output:
        "group.first"
    group:
        "zslurm_smoke_chain"
    resources:
        mem_mb=256,
        runtime=2,
        cpus_per_task=1,
    shell:
        r"""
        test -n "${{ZSLURM_JOB_ID:-}}"
        printf '%s\n' "$ZSLURM_JOB_ID" > {output:q}
        """


rule grouped_second:
    input:
        "group.first"
    output:
        "group.done"
    group:
        "zslurm_smoke_chain"
    resources:
        mem_mb=256,
        runtime=2,
        cpus_per_task=1,
    shell:
        r"""
        test -s {input:q}
        test -n "${{ZSLURM_JOB_ID:-}}"
        test "$(cat {input:q})" = "$ZSLURM_JOB_ID"
        printf 'PASS\t%s\n' "$ZSLURM_JOB_ID" > {output:q}
        """
