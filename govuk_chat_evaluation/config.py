from inspect import isclass
from pathlib import Path
from typing import get_origin, get_args, Optional, Type, TypeVar, Any

import click
import yaml
from pydantic import BaseModel

GenericConfig = TypeVar("GenericConfig", bound="BaseConfig")


class BaseConfig(BaseModel):
    @classmethod
    def apply_click_options(cls, command):
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description

            field_type = field_info.annotation

            if get_origin(field_type) is Optional:
                field_type = get_args(field_type)[0]

            if field_type is bool:
                command = click.option(
                    f"--{field_name}/--no-{field_name}", help=description
                )(command)
            elif (
                # Try avoid complex types such as lists and nested objects
                get_origin(field_type) not in {list, dict}
                and not (isclass(field_type) and issubclass(field_type, BaseModel))
            ):
                command = click.option(f"--{field_name}", help=description)(command)

        return command


def apply_click_options_to_command(config_cls: Type[GenericConfig]):
    def decorator(command):
        return config_cls.apply_click_options(command)

    return decorator


def config_from_cli_args(
    config_path: Path, config_cls: Type[GenericConfig], cli_args: dict[str, Any]
) -> GenericConfig:
    filtered_args = {k: v for k, v in cli_args.items() if v is not None}

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    return config_cls(**(config_data | filtered_args))
