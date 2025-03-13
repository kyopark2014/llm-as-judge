
import utils
import chat

from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

logger = utils.CreateLogger("chat")

CONCISENESS_PROMPT_KOR = """You are an expert data labeler evaluating model outputs for conciseness. 
Answer in Korean.
Your task is to assign a score based on the following rubric:

<Rubric>
  A perfectly concise answer:
  - Contains only the exact information requested.
  - Uses the minimum number of words necessary to convey the complete answer.
  - Omits pleasantries, hedging language, and unnecessary context.
  - Excludes meta-commentary about the answer or the model's capabilities.
  - Avoids redundant information or restatements.
  - Does not include explanations unless explicitly requested.

  When scoring, you should deduct points for:
  - Introductory phrases like "I believe," "I think," or "The answer is."
  - Hedging language like "probably," "likely," or "as far as I know."
  - Unnecessary context or background information.
  - Explanations when not requested.
  - Follow-up questions or offers for more information.
  - Redundant information or restatements.
  - Polite phrases like "hope this helps" or "let me know if you need anything else."
</Rubric>

<Instructions>
  - Carefully read the input and output.
  - Check for any unnecessary elements, particularly those mentioned in the <Rubric> above.
  - The score should reflect how close the response comes to containing only the essential information requested based on the rubric above.
</Instructions>

<Reminder>
  The goal is to reward responses that provide complete answers with absolutely no extraneous information.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""

# llm as judge
conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT_KOR,
    feedback_key="conciseness",
    judge=chat.get_chat(),
)

inputs = "샌프란시스코의 날씨는 어떤가요?" # How is the weather in San Francisco?
outputs = "물어봐 주셔서 감사합니다! 샌프란시스코의 현재 날씨는 맑고 90도입니다." # Thanks for asking! The current weather in San Francisco is sunny and 90 degrees.Thanks for asking! The current weather in San Francisco is sunny and 90 degrees.

eval_result = conciseness_evaluator(
  inputs=inputs,
  outputs=outputs
)

logger.info(f"eval_result: {eval_result}")
logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")
