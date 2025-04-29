import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TypeVar, Type, Any

import yaml
from pydantic import BaseModel

from .config import BaseConfig
import logging

Model = TypeVar("Model", bound=BaseModel)


def project_root() -> Path:
    """Return root directory of this project, used for storing data relative
    to this"""

    return Path(__file__).resolve().parent.parent


def create_output_directory(prefix: str, time: datetime) -> Path:
    """Create a directory for an evaluation based on a directory of the prefix
    and the timestamp this job was run at"""

    time_path = time.replace(microsecond=0).isoformat()

    path = project_root() / "results" / prefix / time_path

    path.mkdir(parents=True)

    relative_path = path.relative_to(project_root())

    logging.info(f"Created output directory at {relative_path}/")

    return path


def jsonl_to_models(file_path: Path, model_class: Type[Model]) -> list[Model]:
    """Open a JSONL file and iterate through the contents, using them to
    hydrate pydantic models"""

    models = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            models.append(model_class(**data))

    return models


def write_generated_to_output(output_dir: Path, generated: list[Model]) -> Path:
    """Write a JSONL file in the output directory that contains the JSON contents
    of each pydantic model in the generated list"""

    output_path = output_dir / "generated.jsonl"
    with open(output_path, "w", encoding="utf8") as file:
        for model in generated:
            file.write(model.model_dump_json() + "\n")

    relative_path = output_path.relative_to(project_root())
    logging.info(f"Wrote generated data to {relative_path}")

    return output_path


def write_config_file_for_reuse(output_dir: Path, config: BaseConfig) -> Path:
    """Write a Config object as a YAML file in the output directory"""
    config_path = output_dir / "config.yaml"
    with open(config_path, "w", encoding="utf8") as file:
        yaml.dump(config.model_dump(mode="json"), file, default_flow_style=False)

    relative_path = config_path.relative_to(project_root())
    logging.info(f"Wrote used config to {relative_path}")

    return config_path


def write_csv_results(
    output_dir: Path,
    data: list[dict[str, Any]],
    filename="results.csv",
    data_label="results",
) -> Path:
    """Take a list of dictionaries and use them to create a CSV file that is
    written to the output directory with the given filename"""
    csv_path = output_dir / filename
    with open(csv_path, "w", encoding="utf8") as file:
        headers = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for record in data:
            writer.writerow(record)

    relative_path = csv_path.relative_to(project_root())
    logging.info(f"Wrote {data_label} to {relative_path}")

    return csv_path
