from pathlib import Path
from dotenv import load_dotenv
import logging

from govuk_chat_evaluation.rag_answers.evaluate import evaluate_and_output_results
from govuk_chat_evaluation.rag_answers.data_models import EvaluationConfig


logger = logging.getLogger(__name__)

load_dotenv(".env")

evaluation_data_path = Path("govuk_chat_evaluation/rag_answers/alessia_test_cases.jsonl")

# Assume this was passed in from a config file or click
raw = {
    "metrics": [
        {"name": "faithfulness", "threshold": 0.8},
        {"name": "bias", "threshold": 0.5}
    ],
    "llm_judge": {
        "model": "gpt-4o",
        "temperature": 0.0
    },
    "n_runs": 3
}

# Assume these are loaded from a config file
deepeval_evaluate_params = {
        "print_results": True,
        "show_indicator": True,
        "max_concurrent": 40,
        "throttle_value": 5,
        "use_cache": False,
        "ignore_errors": True
    }

evaluation_config = EvaluationConfig(**raw)


evaluate_and_output_results(
    _output_dir=Path("results"),
    evaluation_data_path=evaluation_data_path,
    evaluation_config=evaluation_config,
    deepeval_evaluate_params=deepeval_evaluate_params)

