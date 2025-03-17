from pathlib import Path
from typing import cast
from unittest.mock import patch, mock_open

import click
import pytest
from pydantic import BaseModel, Field

from govuk_chat_evaluation.config import (
    BaseConfig,
    apply_click_options_to_command,
    config_from_cli_args,
)


class NestedConfig(BaseModel):
    option: int


class SampleConfig(BaseConfig):
    flag: bool
    option: str = Field(..., description="A string field")


class ComplexConfig(SampleConfig):
    values: list[int] = [1, 2, 3]
    settings: dict[str, str]
    nested: NestedConfig


@pytest.fixture
def mock_yaml_config(mocker):
    yaml_content = """
    flag: true
    option: "test_value"
    """

    mocker.patch("builtins.open", mock_open(read_data=yaml_content))


class TestBaseConfig:
    def test_apply_click_options_creates_options_for_simple_types(self):
        command = click.Command(name="Demo command")
        ComplexConfig.apply_click_options(command)

        options = [option for param in command.params for option in param.opts]
        assert "--flag" in options
        assert "--option" in options
        assert "--values" not in options
        assert "--nested" not in options
        assert "--settings" not in options

    def test_apply_click_options_sets_a_boolean_as_a_flag(self):
        command = click.Command(name="Demo command")
        SampleConfig.apply_click_options(command)

        flag_option = cast(click.core.Option, command.params[0])

        assert flag_option.opts == ["--flag"]
        assert flag_option.is_flag

    def test_apply_click_options_uses_description_as_help(self):
        command = click.Command(name="Demo command")
        SampleConfig.apply_click_options(command)

        flag_option = cast(click.core.Option, command.params[1])

        assert flag_option.opts == ["--option"]
        assert flag_option.help == "A string field"


def test_apply_click_options_to_command():
    with patch.object(SampleConfig, "apply_click_options") as mock_method:
        decorator = apply_click_options_to_command(SampleConfig)

        assert callable(decorator)

        command = click.Command(name="Demo command")
        decorator(command)

        mock_method.assert_called_once_with(command)


@pytest.mark.usefixtures("mock_yaml_config")
def test_config_from_cli_args_uses_config_yaml():
    config = config_from_cli_args(Path("config.yaml"), SampleConfig, {})

    assert type(config) is SampleConfig
    assert config.option == "test_value"
    assert config.flag is True


@pytest.mark.usefixtures("mock_yaml_config")
def test_config_from_cli_args_overrides_config_data_with_cli_args():
    cli_args = {"option": None, "flag": False}
    config = config_from_cli_args(Path("config.yaml"), SampleConfig, cli_args)

    assert config.option == "test_value"
    assert config.flag is False
