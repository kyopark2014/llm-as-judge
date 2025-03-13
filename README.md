# llm-as-judge

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
