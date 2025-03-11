from datetime import datetime
from pydantic import BaseModel
from pathlib import Path
from typing import TypeVar, Type, List
import json

Model = TypeVar("Model", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def create_output_directory(prefix: str, time: datetime) -> Path:
    time_path = time.replace(microsecond=0).isoformat()

    path = project_root() / "results" / prefix / time_path

    path.mkdir(parents=True)

    relative_path = path.relative_to(project_root())

    print(f"Created output directory at {relative_path}/")

    return path


def jsonl_to_models(file_path: str, model_class: Type[Model]) -> List[Model]:
    models = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            models.append(model_class(**data))

    return models


def write_generated_to_output(generated: List[Model], output_dir: Path) -> Path:
    output_file = output_dir / "generated.jsonl"
    with open(output_file, "w", encoding="utf8") as file:
        for model in generated:
            file.write(model.model_dump_json() + "\n")

    relative_path = output_file.relative_to(project_root())
    print(f"Wrote generated data to at {relative_path}")

    return output_file
