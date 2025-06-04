import click
from dotenv import load_dotenv

from . import jailbreak_guardrails
from . import output_guardrails
from . import question_router
from . import rag_answers
from . import retrieval

load_dotenv()


@click.group()
def main():
    """Command line interface to run evaluations of GOV.UK chat"""


main.add_command(jailbreak_guardrails.main)
main.add_command(output_guardrails.main)
main.add_command(question_router.main)
main.add_command(rag_answers.main)
main.add_command(retrieval.main)
