from govuk_chat_evaluation.rag_answers.data_models import EvaluationTestCase


class TestEvaluationTestCase:
    def test_to_llm_test_case(self):
        evaluation_test_case = EvaluationTestCase(
            question="How are you?", ideal_answer="Great", llm_answer="Fine"
        )

        llm_test_case = evaluation_test_case.to_llm_test_case()

        assert llm_test_case.input == evaluation_test_case.question
        assert llm_test_case.expected_output == evaluation_test_case.ideal_answer
        assert llm_test_case.actual_output == evaluation_test_case.llm_answer
