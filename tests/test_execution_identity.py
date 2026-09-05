import pytest

from snakemake_executor_plugin_zslurm import set_execution_identity_env


def test_execution_identity_is_exported_without_forging_the_job_id():
    environment = {"ZSLURM_JOB_ID": "chief-owned"}

    returned = set_execution_identity_env(
        environment,
        instance="lsv-call",
        owner_id="workflow-owner",
        job_url="http://manager:38865/capability",
    )

    assert returned is environment
    assert environment == {
        "ZSLURM_JOB_ID": "chief-owned",
        "ZSLURM_INSTANCE": "lsv-call",
        "ZSLURM_OWNER_ID": "workflow-owner",
        "ZSLURM_JOB_URL": "http://manager:38865/capability",
    }


@pytest.mark.parametrize("field", ["instance", "owner_id", "job_url"])
def test_execution_identity_rejects_missing_fields(field):
    values = {
        "instance": "lsv-call",
        "owner_id": "workflow-owner",
        "job_url": "http://manager:38865/capability",
    }
    values[field] = ""

    with pytest.raises(ValueError, match="must not be empty"):
        set_execution_identity_env({}, **values)
