
import utils
import chat

from openevals.llm import create_llm_as_judge
# from openevals.prompts import CONCISENESS_PROMPT

logger = utils.CreateLogger("chat")

CORRECTNESS_PROMPT_KOR = """You are an expert data labeler evaluating model outputs for correctness. 
Answer in Korean.
Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  When scoring, you should penalize:
  - Factual errors or inaccuracies
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing key information
</Rubric>

<Instructions>
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity
</Instructions>

<Reminder>
  The goal is to evaluate factual correctness and completeness of the response.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

Use the reference outputs below to help you evaluate the correctness of the response:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT_KOR,
    feedback_key="correctness",
    judge=chat.get_chat(),
)

inputs = "지난 1년 동안 doodads의 가격이 얼마나 변했나요?" # How much has the price of doodads changed in the past year?
outputs = "Doodads 가격이 지난 1년 동안 10% 상승했습니다" # Doodads have increased in price by 10% in the past year
reference_outputs = "지난 1년 동안 Doodads 가격이 50% 하락했습니다." # The price of doodads has decreased by 50% in the past year.

eval_result = correctness_evaluator(
  inputs=inputs,
  outputs=outputs,
  reference_outputs=reference_outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")
