class FactualCorrectnessTemplate:
    @staticmethod
    def classify_facts(answer, ground_truth):
        return f"""Given a ground-truth and an answer, analyse each key fact in the answer and classify them in one of the following categories:

- TP (true positive): key facts that are present in both the answer and the ground truth,
- FP (false positive): key facts present in the answer but not found in the ground truth,
- FN (false negative): relevant key facts found in the ground truth but omitted in the answer.

IMPORTANT: Each key fact must be classified in exactly one category. Do not try to interpret the meaning of the ground truth or the answer, just compare the presence of the key facts in them.

You are going to write a JSON to collect your classified key facts into a JSON object. The JSON will have 3 fields, each corresponding to a category: 'TP' (list of true positive key facts), 'FP' (list of false positive key facts), and 'FN' (list of false negative key facts).

Now consider the following python pydantic BaseModel for the JSON schema:

class ClassifiedFacts(BaseModel):
    TP: list[str]
    FP: list[str]
    FN: list[str]

class FactClassificationResult(BaseModel):
   classified_facts: ClassifiedFacts

IMPORTANT: Write your output according to the FactClassificationResult schema. On the output, include only the JSON.

**
Here are three examples.

Example Answer: "Universal Credit is a monthly payment for living costs. It replaces benefits like Child Tax Credit and Job Allowance. If you get a Migration Notice, you must transition to Universal Credit within 3 months."
Example Ground Truth: "Universal Credit is a payment to help with your living costs. It’s paid monthly. You may be able to get it if you’re on a low income or out of work. Universal Credit is replacing Child Tax Credit, Housing Benefit, and Income Support. If you get a Migration Notice, you must move to Universal Credit within 3 months to keep getting financial support."
Example output JSON:
{{
   "classified_facts": {{
       "TP": [
           "Universal Credit is a monthly payment for living costs",
           "It replaces benefits like Child Tax Credit",
           "If you get a Migration Notice, you must transition to Universal Credit within 3 months"
       ],
       "FP": [
           "It replaces benefits like Job Allowance"
       ],
       "FN": [
           "You may be able to get it if you’re on a low income or out of work",
           "Universal Credit is replacing Housing Benefit and Income Support"
       ]
   }}
}}

Example Answer: "The sun is powered by nuclear fission, similar to nuclear reactors on Earth. Its primary function is to provide light which is essential to Earth's climate system."
Example Ground Truth: "The sun is powered by nuclear fusion. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents."
Example output JSON:
{{
    "classified_facts": {{
                    "TP": [
                        "The sun's primary function is to provide light",
                        "The sun's light is essential to Earth's climate system",
                        ],
                    "FP": [
                        "The sun is powered by nuclear fission",
                        "similar to nuclear reactors on Earth",
                    ],
                    "FN": [
                        "The sun is powered by nuclear fusion",
                        "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy",
                        "This energy provides heat and light, essential for life on Earth",
                        "The sun helps to drive the weather and ocean currents",
                    ],
                }}
}}

Example Answer: "The boiling point of water is 100 degrees Celsius at sea level."
Example Ground Truth: "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude."
Example output JSON:
{{
    "classified_facts": {{
                    "TP": [
                        "The boiling point of water is 100 degrees Celsius at sea level"
                        ],
                    "FP": [],
                    "FN": [
                        "The boiling point can change with altitude",
                        "The boiling point of water is 212 degrees Fahrenheit at sea level",
                        ],
                }}
}}

**

Ground Truth:
{ground_truth}

Actual Output:
{answer}

JSON:
"""
