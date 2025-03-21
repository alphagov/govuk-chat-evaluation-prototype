import click

from . import jailbreak_guardrails
from . import rag_answers


@click.group()
def main():
    """Command line interface to run evaluations of GOV.UK chat"""


main.add_command(jailbreak_guardrails.main)
main.add_command(rag_answers.main)
