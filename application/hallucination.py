
import utils
import chat

from openevals.llm import create_llm_as_judge
from openevals.prompts import HALLUCINATION_PROMPT

logger = utils.CreateLogger("chat")

HALLUCINATION_PROMPT_KOR = """You are an expert data labeler evaluating model outputs for hallucinations. 
Answer in Korean.
Your task is to assign a score based on the following rubric:

<Rubric>
  A response without hallucinations:
  - Contains only verifiable facts that are directly supported by the input context
  - Makes no unsupported claims or assumptions
  - Does not add speculative or imagined details
  - Maintains perfect accuracy in dates, numbers, and specific details
  - Appropriately indicates uncertainty when information is incomplete
</Rubric>

<Instructions>
  - Read the input context thoroughly
  - Identify all claims made in the output
  - Cross-reference each claim with the input context
  - Note any unsupported or contradictory information
  - Consider the severity and quantity of hallucinations
</Instructions>

<Reminder>
  Focus solely on factual accuracy and support from the input context. Do not consider style, grammar, or presentation in scoring. A shorter, factual response should score higher than a longer response with unsupported claims.
</Reminder>

Use the following context to help you evaluate for hallucinations in the output:

<context>
{context}
</context>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

If available, you may also use the reference outputs below to help you identify hallucinations in the response:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""

hallucination_evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT_KOR,
    feedback_key="hallucination_",
    judge=chat.get_chat(),
)

inputs = "React는 무엇인가요?" # What is a doodad?
outputs = "React는 사용자 인터페이스(UI)를 렌더링하기 위한 JavaScript 라이브러리입니다. UI는 버튼, 텍스트, 이미지와 같은 작은 요소로 구성됩니다. "
context = """
ReAct(Reasoning and Acting)는 대규모 언어 모델(LLM)의 추론 능력과 행동 능력을 결합한 AI 프레임워크입니다. 제가 찾은 정보를 바탕으로 설명해 드리겠습니다.

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
ReAct는 LangChain과 같은 프레임워크에서도 구현되어 있어, 언어 모델과 다양한 도구를 결합하여 복잡한 작업을 수행하는 에이전트를 만드는 데 활용되고 있습니다.
"""
reference_outputs = ""

eval_result = hallucination_evaluator(
  context=context,
  inputs=inputs,
  outputs=outputs,
  reference_outputs=reference_outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")
