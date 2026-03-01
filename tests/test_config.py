"""Tests for KairoScale.config."""

import tempfile
from pathlib import Path

import pytest
import yaml

from KairoScale.config import load_config, load_yaml_config


def test_loadConfigFromCliArgs(tmp_path):
    args = {
        "repo_path": str(tmp_path),
        "entry_point": "main.py",
        "provider": "openai",
        "verbose": True,
    }
    config = load_config(args)
    assert config.entry_point == "main.py"
    assert config.provider == "openai"
    assert config.verbose is True


def test_loadConfigNoneValuesIgnored(tmp_path):
    args = {
        "repo_path": str(tmp_path),
        "entry_point": None,
        "provider": None,
    }
    config = load_config(args)
    # Defaults should apply when CLI args are None
    assert config.entry_point == "train.py"
    assert config.provider == "claude"


def test_loadConfigYamlMerge(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml.dump({
        "entry_point": "run.py",
        "provider": "openai",
        "profile_steps": 50,
    }))

    args = {"repo_path": str(tmp_path), "provider": "claude"}
    config = load_config(args, yaml_file)
    # CLI overrides YAML
    assert config.provider == "claude"
    # YAML value used when not in CLI
    assert config.entry_point == "run.py"
    assert config.profile_steps == 50


def test_loadConfigMissingRepoPath():
    with pytest.raises(ValueError, match="repo_path"):
        load_config({})


def test_loadYamlConfig(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml.dump({"gpu_type": "h100", "top_k": 3}))
    result = load_yaml_config(yaml_file)
    assert result["gpu_type"] == "h100"
    assert result["top_k"] == 3
