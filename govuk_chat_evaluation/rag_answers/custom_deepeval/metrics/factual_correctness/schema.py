from pydantic import BaseModel


class ClassifiedFacts(BaseModel):
    TP: list[str] = []
    FP: list[str] = []
    FN: list[str] = []

    def has_facts(self) -> bool:
        return bool(self.TP or self.FP or self.FN)


class FactClassificationResult(BaseModel):
    classified_facts: ClassifiedFacts
