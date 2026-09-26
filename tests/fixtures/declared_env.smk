"""Worker-launch smoke test; writes only a small JSON file in its workdir."""

import json
import os
from pathlib import Path

envvars:
    "SHORT_READ_RESTART_MANIFEST",
    "ZSLURM_TEST_DECLARED",
    "ZSLURM_TEST_PROVIDER"


rule env_probe:
    output:
        "env-check.json"
    resources:
        mem_mb=256,
        time=120,
        n="0.1",
        ssd_use="no"
    run:
        values = {
            key: os.environ.get(key)
            for key in (
                "SHORT_READ_RESTART_MANIFEST",
                "ZSLURM_TEST_DECLARED",
                "ZSLURM_TEST_PROVIDER",
                "ZSLURM_INSTANCE",
                "ZSLURM_JOB_ID",
                "OMP_NUM_THREADS",
                "SNAKEMAKE_PROFILE",
            )
        }
        manifest = Path(values["SHORT_READ_RESTART_MANIFEST"])
        values["reused_samples"] = len(json.loads(manifest.read_text())["reused_samples"])
        Path(output[0]).write_text(json.dumps(values, sort_keys=True) + "\n")
