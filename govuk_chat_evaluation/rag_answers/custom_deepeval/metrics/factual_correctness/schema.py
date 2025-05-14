from pydantic import BaseModel


class ClassifiedFacts(BaseModel):
    TP: list[str]
    FP: list[str]
    FN: list[str]


class FactClassificationResult(BaseModel):
    classified_facts: ClassifiedFacts
