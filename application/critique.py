
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
    feedback_key="critique",
    judge=chat.get_chat(),
)

outputs = """ReAct(Reasoning and Acting)는 대규모 언어 모델(LLM)의 추론 능력과 행동 능력을 결합한 AI 프레임워크입니다. 제가 찾은 정보를 바탕으로 설명해 드리겠습니다.

ReAct의 주요 특징:

추론과 행동의 통합: 기존에는 언어 모델의 추론 능력(예: chain-of-thought 프롬프팅)과 행동 능력(예: 행동 계획 생성)이 별개로 연구되었지만, ReAct는 이 두 가지를 함께 활용합니다.

작동 방식:

언어 모델에게 추론 과정을 언어로 표현하도록 유도하면서 동시에 행동을 취하게 합니다.
모델이 생각하는 과정을 텍스트로 표현하고, 그에 따른 행동을 취한 후, 환경으로부터 피드백을 받아 다시 추론하는 과정을 반복합니다.
장점:

환상(hallucination)과 오류 전파 문제를 줄일 수 있습니다.
인간이 이해하기 쉬운 문제 해결 과정을 생성합니다.
복잡한 추론이 필요한 작업에서 더 나은 성능을 보입니다.
적용 분야:

지식 집약적 추론 작업(예: HotpotQA, Fever)
웹 탐색, 텍스트 게임, 로봇 제어 등 다양한 상호작용 환경
ReAct는 LangChain과 같은 프레임워크에서도 구현되어 있어, 언어 모델과 다양한 도구를 결합하여 복잡한 작업을 수행하는 에이전트를 만드는 데 활용되고 있습니다."""

eval_result = critique_evaluator(
  inputs=None,
  outputs=outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")


