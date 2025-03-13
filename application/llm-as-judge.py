
import utils
import chat

from openevals.llm import create_llm_as_judge
from langgraph_reflection import create_reflection_graph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

logger = utils.CreateLogger("chat")

debugMode = False

# assistant graph
def assistant_graph():
    def call_model(state):
        """Process the user query with a large language model."""
        model = chat.get_chat()

        return {"messages": model.invoke(state["messages"])}

    def buildAssistantAgent():
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", call_model)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        return workflow.compile()

    app = buildAssistantAgent()
    return app

# test of assistant_graph
if debugMode:
    query = "ReAct Agent란 무엇인가요?"
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}

    assistant = assistant_graph()
    output = assistant.invoke({"messages": inputs}, config)    

    assistant_output = output["messages"][-1].content
    logger.info(f"assistant output: {assistant_output}")

# judge graph    
def judge_graph():
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

    def judge_node(state, config):
        """Evaluate the assistant's response using a separate judge model."""

        critique_evaluator = create_llm_as_judge(
            prompt=CRITIQUE_PROMPT_KOR,
            feedback_key="critique",
            judge=chat.get_chat(),
        )
        
        eval_result = critique_evaluator(outputs=state["messages"][-1].content, inputs=None)
        logger.info(f"score: {eval_result.get('score')}")
        logger.info(f"comment: {eval_result.get('comment')}")

        if eval_result["score"]:
            logger.info(f"Response approved by judge")
            return
        else:
            # Otherwise, return the judge's critique as a new user message
            logger.info("Judge requested improvements")
            return {"messages": [{"role": "user", "content": eval_result["comment"]}]}
        
    def buildJudgeAgent():
        workflow = StateGraph(MessagesState)

        workflow.add_node("judge", judge_node)
        workflow.add_edge(START, "judge")
        workflow.add_edge("judge", END)

        return workflow.compile()

    app = buildJudgeAgent()
    return app

# test of judge_graph
if debugMode:
    inputs = [HumanMessage(content=assistant_output)]
    config = {"recursion_limit": 50}

    judge = judge_graph()
    output = judge.invoke({"messages": inputs}, config)
    logger.info(f"judge output: {output["messages"][-1].content}")

def create_graphs():
    return create_reflection_graph(assistant_graph, judge_graph).compile()

reflection_app = create_graphs()

from typing import TypedDict
class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""
    finish: bool

if __name__ == "__main__":
    example_query = [
        {
            "role": "user",
            "content": "Explain how nuclear fusion works and why it's important for clean energy",
        }
    ]

    # Process the query through the reflection system
    logger.info("Running example with reflection...")
    result = reflection_app.invoke({"messages": example_query})
    logger.info(f"result: {result}")

    # query = "Explain how nuclear fusion works and why it's important for clean energy"
    # inputs = [HumanMessage(content=query)]
    # config = {"recursion_limit": 50}

    # output = reflection_app.invoke({"messages": inputs}, config)    
    # logger.info(f"output: {output}")