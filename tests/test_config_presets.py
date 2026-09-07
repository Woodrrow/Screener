from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from screener.config import ScreenConfig, load_screen_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "screen.yaml"


def test_shipped_config_is_valid() -> None:
    config = load_screen_config(REPO_CONFIG)
    assert sorted(config.presets) == ["balanced", "momentum", "value"]
    assert config.default_preset == "balanced"


def test_every_shipped_preset_sums_to_one() -> None:
    config = load_screen_config(REPO_CONFIG)
    for name, weights in config.presets.items():
        assert sum(weights.values()) == pytest.approx(1.0), name


def test_unknown_factor_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown factors"):
        ScreenConfig.model_validate({"presets": {"p": {"momentum_9000d": 1.0}}})


def test_weights_that_do_not_sum_to_one_are_rejected() -> None:
    with pytest.raises(ValidationError, match="sum to"):
        ScreenConfig.model_validate(
            {"presets": {"p": {"momentum_7d": 0.5, "liquidity": 0.2}}, "default_preset": "p"}
        )


def test_negative_weights_are_rejected_with_the_reason() -> None:
    with pytest.raises(ValidationError, match="LOWER_IS_BETTER"):
        ScreenConfig.model_validate(
            {
                "presets": {"p": {"momentum_7d": 1.2, "volatility": -0.2}},
                "default_preset": "p",
            }
        )


def test_empty_preset_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no weights"):
        ScreenConfig.model_validate({"presets": {"p": {}}, "default_preset": "p"})


def test_default_preset_must_exist() -> None:
    with pytest.raises(ValidationError, match="default_preset"):
        ScreenConfig.model_validate(
            {"presets": {"p": {"momentum_7d": 1.0}}, "default_preset": "nope"}
        )


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ScreenConfig.model_validate({"presetz": {}})


def test_weights_for_falls_back_to_the_default() -> None:
    config = load_screen_config(REPO_CONFIG)
    assert config.weights_for(None)[0] == "balanced"
    assert config.weights_for("value")[0] == "value"
    with pytest.raises(KeyError, match="unknown preset"):
        config.weights_for("aggressive")
