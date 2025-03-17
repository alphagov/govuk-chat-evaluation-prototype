import csv
import json
from datetime import datetime
from pathlib import Path, PosixPath
from pydantic import BaseModel

import pytest
import yaml

from govuk_chat_evaluation.config import BaseConfig
from govuk_chat_evaluation.file_system import (
    project_root,
    create_output_directory,
    jsonl_to_models,
    write_generated_to_output,
    write_config_file_for_reuse,
    write_csv_results,
)


class SampleConfig(BaseConfig):
    what: str


class SampleModel(BaseModel):
    name: str
    age: int


@pytest.fixture
def sample_jsonl(tmp_path):
    file_path = tmp_path / "sample.jsonl"
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")
    return file_path


def test_project_root():
    assert project_root().exists()
    assert Path(__file__).is_relative_to(project_root())


def test_create_output_directory(mock_project_root):
    time = datetime(2023, 2, 4, 8, 30, 0)
    output_dir = create_output_directory("test_prefix", time)
    assert output_dir.exists()
    assert output_dir.relative_to(mock_project_root) == PosixPath(
        "results/test_prefix/2023-02-04T08:30:00"
    )


def test_jsonl_to_models(sample_jsonl):
    models = jsonl_to_models(sample_jsonl, SampleModel)
    assert len(models) == 2
    assert models[0].name == "Alice"
    assert models[1].age == 25


def test_write_generated_to_output(mock_project_root):
    models = [SampleModel(name="Alice", age=30), SampleModel(name="Bob", age=25)]
    output_path = write_generated_to_output(mock_project_root, models)
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    assert len(lines) == 2


def test_write_config_file_for_reuse(mock_project_root):
    config = SampleConfig(what="Testing config")
    config_path = write_config_file_for_reuse(mock_project_root, config)
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    assert content == {"what": "Testing config"}


def test_write_csv_results(mock_project_root):
    data = [{"col1": "val1", "col2": "val2"}, {"col1": "val3", "col2": "val4"}]
    csv_path = write_csv_results(mock_project_root, data)

    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["col1"] == "val1"
