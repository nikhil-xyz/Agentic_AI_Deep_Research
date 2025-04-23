from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class LLMNode:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, state:State):
        return {"messages": [self.llm.invoke(state['messages'])]}