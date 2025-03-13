
import utils
import chat

from openevals.llm import create_llm_as_judge
# from openevals.prompts import CONCISENESS_PROMPT

logger = utils.CreateLogger("chat")

CRITIQUE_PROMPT_KOR = """You are an expert judge evaluating AI responses. 
Answer in Korean.
Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?

If the response meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve.

<response>
{outputs}
</response>"""

critique_evaluator = create_llm_as_judge(
    prompt=CRITIQUE_PROMPT_KOR,
    feedback_key="critiqueness",
    judge=chat.get_chat(),
)

outputs = "지난 1년 동안 doodads의 가격이 얼마나 변했나요?" 

eval_result = critique_evaluator(
  outputs=outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")


