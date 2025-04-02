from pydantic import BaseModel


# I expect this model to be a subclass of the model used for generating answers
class EvaluationTestCase(BaseModel):
    question: str
    ideal_answer: str
    llm_answer: str
    # TODO: lots more data fields
