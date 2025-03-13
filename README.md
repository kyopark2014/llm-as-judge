# LLM으로 평가하기

여기에서는 LangChain의 [openevals](https://github.com/langchain-ai/openevals)을 이용해 LLM으로 평가하는 방법에 대해 설명합니다.

## OpenEvals 

[openevals](https://github.com/langchain-ai/openevals)는 LangChain의 오픈소스 프로젝트로서 LLM으로 평가하는 프로세스를 보여주고 있습니다.

[Evaluating LLMs with OpenEvals](https://www.youtube.com/watch?v=J-F30jRyhoA)에서는 평가하는것을 아래와 같이 데모로 보여줍니다.

![image](https://github.com/user-attachments/assets/4d693c12-79fc-4508-8b2a-2333310c13d2)


이때 평가과정은 아래와 같이 inputs와 referenceOutputs를 놓고 LLM으로 평가합니다. 

![image](https://github.com/user-attachments/assets/e1ee6eab-7f61-41a5-b661-4f4f192f82cf)

### 사전 준비 

openevals를 설치합니다.

```text
pip install openevals
```

## 구현 예제

### Conciseness 

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


### Correctness

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

### Hallucination

[hallucination.py](./application/hallucination.py)와 같이 Hallucination을 확인하기 위한 evaluator를 이용할 수 있습니다.

```python
hallucination_evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT_KOR,
    feedback_key="hallucination_",
    judge=chat.get_chat(),
)

inputs = "React는 무엇인가요?" 
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
```

이때의 결과는 아래와 같습니다.

```python
hallucination.py:95 | score: False
hallucination.py:96 | comment: 주어진 출력을 평가하기 위해 입력 컨텍스트와 비교해 보겠습니다.

출력에서는 "React는 사용자 인터페이스(UI)를 렌더링하기 위한 JavaScript 라이브러리입니다. UI는 버튼, 텍스트, 이미지와 같은 작은 요소로 구성됩니다."라고 설명하고 있습니다.

그러나 입력 컨텍스트에서는 "ReAct(Reasoning and Acting)"에 대해 설명하고 있으며, 이는 대규모 언어 모델(LLM)의 추론 능력과 행동 능력을 결합한 AI 프레임워크라고 명시되어 있습니다. 

출력은 완전히 다른 기술에 대해 설명하고 있습니다:
1. 출력은 "React" JavaScript 라이브러리에 대해 설명하고 있습니다.
2. 입력 컨텍스트는 "ReAct" AI 프레임워크에 대한 것입니다.

이는 심각한 환각(hallucination)으로, 출력이 제공된 컨텍스트와 전혀 관련이 없는 정보를 제시하고 있습니다. 출력은 컨텍스트에서 제공된 어떤 정보도 정확하게 반영하지 않고 있으며, 완전히 다른 기술에 대해 설명하고 있습니다.

따라서, 점수는 false여야 합니다.
```

### Critique

[critique.py](./application/critique.py)와 같이 비평을 수행할 수 있습니다.

```python
critique_evaluator = create_llm_as_judge(
    prompt=CRITIQUE_PROMPT_KOR,
    feedback_key="critique",
    judge=chat.get_chat(),
)
eval_result = critique_evaluator(
  inputs=None,
  outputs=outputs
)

logger.info(f"score: {eval_result.get('score')}")
logger.info(f"comment: {eval_result.get('comment')}")
```

이때의 결과는 아래와 같습니다.

```text
critique.py:63 | score: True
critique.py:64 | comment: AI 응답에 대한 평가를 한국어로 진행하겠습니다.

1. 정확성: 응답은 ReAct(Reasoning and Acting) 프레임워크에 대한 정확한 정보를 제공하고 있습니다. ReAct가 추론(reasoning)과 행동(acting)을 결합한 프레임워크라는 점, 작동 방식, 장점 등이 정확하게 설명되어 있습니다.

2. 완전성: 응답은 ReAct의 정의, 주요 특징, 작동 방식, 장점, 적용 분야 등을 포괄적으로 다루고 있어 사용자의 질문에 충분히 답변하고 있습니다.

3. 명확성: 응답은 명확한 구조로 정리되어 있으며, 각 섹션이 잘 구분되어 있습니다. 전문적인 개념을 이해하기 쉽게 설명하고 있습니다.

4. 유용성: 응답은 ReAct 프레임워크에 대한 실용적인 정보를 제공하며, 적용 분야와 장점을 설명함으로써 사용자가 이 기술을 어떻게 활용할 수 있는지 이해하는 데 도움이 됩니다.

5. 안전성: 응답에는 유해하거나 부적절한 내용이 포함되어 있지 않습니다.

모든 평가 기준에서 AI의 응답은 만족스러운 수준을 보여주고 있습니다. 정보가 정확하고, 질문에 완전히 답변하며, 명확하게 구조화되어 있고, 유용한 정보를 제공하며, 안전한 내용을 담고 있습니다. 따라서, 점수는 True가 되어야 합니다.
```

### JSON evaluator

[json_match_evaluation.py](./application/json_match_evaluation.py)와 같이 json을 평가할 수 있습니다.

```python
evaluator = create_json_match_evaluator(
    # How to aggregate feedback keys in each element of the list: "average", "all", or None
    # "average" returns the average score. "all" returns 1 only if all keys score 1; otherwise, it returns 0. None returns individual feedback chips for each key
    aggregator="average",
    # Remove if evaluating a single structured output. This aggregates the feedback keys across elements of the list. Can be "average" or "all". Defaults to "all". "all" returns 1 if each element of the list is 1; if any score is not 1, it returns 0. "average" returns the average of the scores from each element. 
    list_aggregator="all",
    rubric={
        "a": "Does the answer mention all the fruits in the reference answer?"
    },
    judge=chat.get_chat(),
    use_reasoning=False    
)

eval_result = evaluator(
    outputs=outputs, 
    reference_outputs=reference_outputs
)
logger.info(f"eval_result: {eval_result}")
```

이때의 결과는 아래와 같습니다.

```text
json_match_evaluation.py:35 | eval_result: [{'key': 'json_match:average', 'score': 0, 'comment': None}]
```

## To-Do: LLM As a Judge

[llm-as-judge.py](./application/llm-as-judge.py)에서는 [LangGraph-Reflection](https://github.com/langchain-ai/langgraph-reflection/tree/main)을 이용하여 reflection 패턴을 multi agent로 구현합니다.  

아래와 같이 langgraph-reflection을 설치합니다. 

```text
pip install langgraph-reflection
```

[LLM-as-a-Judge](https://github.com/langchain-ai/langgraph-reflection/blob/main/examples/llm_as_a_judge.py)를 참조하여 [llm-as-judge.py](./application/llm-as-judge.py)을 구현한 결과 아래와 같이 compile 에러가 발생하고 있어 debug 예정입니다.

```text
Traceback (most recent call last):
  File "/Users/ksdyb/Documents/src/llm-as-judge/application/llm-as-judge.py", line 114, in <module>
    reflection_app = create_graphs()
  File "/Users/ksdyb/Documents/src/llm-as-judge/application/llm-as-judge.py", line 112, in create_graphs
    return create_reflection_graph(assistant_graph, judge_graph).compile()
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.13/site-packages/langgraph_reflection/__init__.py", line 30, in create_reflection_graph
    _state_schema = state_schema or graph.builder.schema
                                    ^^^^^^^^^^^^^
```
