from pathlib import Path

import pytest

from gdm_factor_diffusion.common.config import (
    get_config_value,
    load_config,
    resolve_config_path,
)


def test_config_deep_merge_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base.yaml"
    update = tmp_path / "update.yaml"
    base.write_text(
        "runtime:\n  seed: 1\n  device: cpu\npaths:\n  root: ${TEST_ROOT}\n",
        encoding="utf-8",
    )
    update.write_text("runtime:\n  device: cuda\n", encoding="utf-8")
    monkeypatch.setenv("TEST_ROOT", str(tmp_path))

    config = load_config(
        [base, update],
        overrides=["runtime.seed=7", "model.hidden=64"],
    )

    assert get_config_value(config, "runtime.seed") == 7
    assert get_config_value(config, "runtime.device") == "cuda"
    assert get_config_value(config, "model.hidden") == 64
    assert get_config_value(config, "paths.root") == str(tmp_path)


def test_missing_config_value_is_explicit() -> None:
    with pytest.raises(KeyError, match="missing.value"):
        get_config_value({}, "missing.value")


def test_archived_pre_stage3_config_resolves_only_when_active_path_is_missing() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    requested = implementation_root / "configs" / "phase6d_c_sealed_campaign.yaml"
    resolved = resolve_config_path(requested)

    assert not requested.exists()
    assert resolved == (
        implementation_root
        / "history"
        / "pre_stage3_2026-06-21"
        / "configs"
        / requested.name
    )
    assert load_config(requested)["campaign"]["name"]
