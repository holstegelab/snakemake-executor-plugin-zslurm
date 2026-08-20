import pytest

from snakemake_executor_plugin_zslurm import (
    DEFAULT_PRIORITY,
    set_dcache_slot_env,
    set_dcache_transfer_slot_env,
    submit_args_with_priority,
)


def test_transfer_slot_resource_is_forwarded_without_inheritance():
    env = {"ZSLURM_DCACHE_TRANSFER_SLOTS": "99"}
    assert set_dcache_transfer_slot_env(env, 1) == 1.0
    assert env["ZSLURM_DCACHE_TRANSFER_SLOTS"] == "1.0"
    assert set_dcache_transfer_slot_env(env, 0) == 0.0
    assert "ZSLURM_DCACHE_TRANSFER_SLOTS" not in env


def test_negative_transfer_slots_are_rejected():
    with pytest.raises(ValueError):
        set_dcache_transfer_slot_env({}, -1)


def test_directional_transfer_slots_are_independent_and_clear_inherited_env():
    env = {
        "ZSLURM_DCACHE_DOWNLOAD_SLOTS": "9",
        "ZSLURM_DCACHE_UPLOAD_SLOTS": "8",
        "ZSLURM_DCACHE_TRANSFER_SLOTS": "7",
    }
    values = set_dcache_slot_env(env, download=2, upload=3)
    assert values == {
        "download": 2.0,
        "upload": 3.0,
        "legacy": 0.0,
        "s3_download": 0.0,
    }
    assert env == {
        "ZSLURM_DCACHE_DOWNLOAD_SLOTS": "2.0",
        "ZSLURM_DCACHE_UPLOAD_SLOTS": "3.0",
        "ZSLURM_DCACHE_TRANSFER_SLOTS": "3.0",
    }


def test_directional_and_legacy_slots_cannot_be_mixed():
    with pytest.raises(ValueError, match="cannot be combined"):
        set_dcache_slot_env({}, download=1, legacy=1)


def test_s3_download_has_native_and_rolling_upgrade_fallback_metadata():
    env = {
        "ZSLURM_S3_DOWNLOAD_SLOTS": "9",
        "ZSLURM_DCACHE_DOWNLOAD_SLOTS": "8",
        "ZSLURM_DCACHE_UPLOAD_SLOTS": "7",
        "ZSLURM_DCACHE_TRANSFER_SLOTS": "6",
    }
    values = set_dcache_slot_env(env, s3_download=2)
    assert values == {
        "download": 0.0,
        "upload": 0.0,
        "legacy": 0.0,
        "s3_download": 2.0,
    }
    assert env == {
        "ZSLURM_S3_DOWNLOAD_SLOTS": "2.0",
        "ZSLURM_DCACHE_DOWNLOAD_SLOTS": "2.0",
        "ZSLURM_DCACHE_TRANSFER_SLOTS": "2.0",
    }


def test_s3_and_dcache_slots_cannot_be_mixed():
    with pytest.raises(ValueError, match="cannot be combined"):
        set_dcache_slot_env({}, download=1, s3_download=1)


def test_default_priority_keeps_legacy_submit_signature():
    original = ["job", "owner"]
    assert DEFAULT_PRIORITY == 100
    assert submit_args_with_priority(original, 100) == original


def test_nondefault_priority_appends_idempotency_placeholder_and_priority():
    assert submit_args_with_priority(["job", "owner"], 200) == [
        "job",
        "owner",
        None,
        200,
    ]


def test_zero_priority_is_supported():
    assert submit_args_with_priority(["job", "owner"], 0)[-2:] == [None, 0]


def test_negative_priority_is_supported():
    assert submit_args_with_priority(["job", "owner"], -4)[-2:] == [None, -4]


def test_invalid_priority_is_rejected():
    with pytest.raises(ValueError):
        submit_args_with_priority([], "urgent")
