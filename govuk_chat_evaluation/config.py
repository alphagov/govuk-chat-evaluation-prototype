from argparse import ArgumentParser, BooleanOptionalAction
from inspect import isclass
from typing import get_origin, get_args, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

GenericConfig = TypeVar("GenericConfig", bound="BaseConfig")


class BaseConfig(BaseModel):
    @classmethod
    def apply_cli_args(cls, parser: ArgumentParser):
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description

            field_type = field_info.annotation

            if get_origin(field_type) is Optional:
                field_type = get_args(field_type)[0]

            if field_type is bool:
                parser.add_argument(
                    f"--{field_name}", help=description, action=BooleanOptionalAction
                )
            elif (
                # Try avoid complex types such as lists and nested objects
                get_origin(field_type) not in {list, dict}
                and not (isclass(field_type) and issubclass(field_type, BaseModel))
            ):
                parser.add_argument(f"--{field_name}", help=description)


def config_from_cli_args(
    parser: ArgumentParser, default_config_path: str, config_cls: Type[GenericConfig]
) -> GenericConfig:
    parser.add_argument(
        "--config_file",
        default=default_config_path,
        help="The config file to use for this evaluation",
    )
    config_cls.apply_cli_args(parser)

    args = vars(parser.parse_args())
    filtered_args = {k: v for k, v in args.items() if v is not None}

    config_file_path = filtered_args["config_file"]

    with open(config_file_path, "r") as file:
        config_data = yaml.safe_load(file)

    return config_cls(**(config_data | filtered_args))
