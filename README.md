# llm-as-judge

## OpenEvals 

[openevals](https://github.com/langchain-ai/openevals)는 LangChain의 오픈소스 프로젝트로서 LLM으로 평가하는 프로세스를 보여주고 있습니다. 기존의 테스트보다 훨씬 우수한 결과를 보여줍니다. 

[Evaluating LLMs with OpenEvals](https://www.youtube.com/watch?v=J-F30jRyhoA)에서는 평가하는것을 아래와 같이 데모로 보여줍니다.

![image](https://github.com/user-attachments/assets/4d693c12-79fc-4508-8b2a-2333310c13d2)


이때 평가과정은 아래와 같이 inputs와 referenceOutputs를 놓고 LLM으로 평가합니다. 

![image](https://github.com/user-attachments/assets/e1ee6eab-7f61-41a5-b661-4f4f192f82cf)

평가하는 LLM은 아래와 같이 정의합니다.

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from langchain_anthropic import ChatAnthropic

anthropic_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.5),
)
```

openevals를 설치합니다.

```text
pip install openevals
```

## Conciseness 

[conciseness.py](./application/conciseness.py)와 같이 CONCISENESS_PROMPT을 이용합니다.

```python
conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT_KOR,
    feedback_key="conciseness",
    judge=chat.get_chat(),
)

inputs = "샌프란시스코의 날씨는 어떤가요?" 
outputs = "물어봐 주셔서 감사합니다! 샌프란시스코의 현재 날씨는 맑고 90도입니다." 

eval_result = conciseness_evaluator(
  inputs=inputs,
  outputs=outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")
```

이때의 결과는 아래와 같습니다.

```text
conciseness.py:68 | score: False
conciseness.py:69 | comment: 이 응답을 분석해 보겠습니다:

1. "물어봐 주셔서 감사합니다!" - 이것은 불필요한 예의 표현으로, 완벽하게 간결한 답변에서는 제외되어야 합니다.

2. "샌프란시스코의 현재 날씨는 맑고 90도입니다." - 이 부분은 질문에 대한 직접적인 답변이지만, "현재"라는 단어는 질문에서 명시적으로 요구되지 않았으므로 엄격히 말하면 불필요합니다.

완벽하게 간결한 답변은 "샌프란시스코 날씨는 맑고 90도입니다."와 같이 되어야 합니다.

응답에는 불필요한 예의 표현과 약간의 추가 단어가 포함되어 있습니다. 이는 루브릭에서 언급된 "예의 표현"과 "불필요한 단어"에 해당합니다.

따라서, 이 응답은 완벽하게 간결하지 않습니다. 

Thus, the score should be: false.
```


## Correctness

[correctness.py](./application/correctness.py)와 같이 구현할 수 있습니다.

```python
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT_KOR,
    feedback_key="correctness",
    judge=chat.get_chat(),
)

inputs = "지난 1년 동안 doodads의 가격이 얼마나 변했나요?" 
outputs = "Doodads 가격이 지난 1년 동안 10% 상승했습니다" 
reference_outputs = "지난 1년 동안 Doodads 가격이 50% 하락했습니다." 

eval_result = correctness_evaluator(
  inputs=inputs,
  outputs=outputs,
  reference_outputs=reference_outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")
```

이때의 결과는 아래와 같습니다.

```text
correctness.py:72 | score: False
correctness.py:73 | comment: 주어진 출력과 참조 출력을 비교해 보겠습니다.

출력: "Doodads 가격이 지난 1년 동안 10% 상승했습니다"
참조 출력: "지난 1년 동안 Doodads 가격이 50% 하락했습니다."

이 두 진술은 명백히 모순됩니다:
1. 출력은 doodads 가격이 10% 상승했다고 주장합니다.
2. 참조 출력은 doodads 가격이 50% 하락했다고 주장합니다.

이는 완전히 반대되는 정보로, 가격 변동의 방향(상승 vs 하락)과 크기(10% vs 50%)가 모두 다릅니다. 이는 심각한 사실적 오류입니다.

채점 기준에 따르면, 정확하고 완전한 정보를 제공하고 사실적 오류가 없어야 합니다. 그러나 주어진 출력은 참조 출력과 비교했을 때 명백한 사실적 오류를 포함하고 있습니다.

따라서, 이 출력은 정확성 측면에서 기준을 충족하지 못합니다. 가격이 상승했다고 주장하는 것은 실제로 하락했다는 사실과 모순되며, 변동 폭도 크게 다릅니다.

Thus, the score should be: false.
```




