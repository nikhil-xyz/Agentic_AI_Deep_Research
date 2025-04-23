# Dependencies
import os
import streamlit as st

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END
from utilities import State, LLMNode


# Load environment variables from .env file
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set the page configuration
st.title("Agentic System")
question = st.text_input("Mention your question here")

# Setting up the tool (Tavily) to use it during the data collection 
tavily = TavilySearchResults(
    include_image=True
)

# Setting up the LLM (Groq) 
llm = ChatGroq(model='qwen-qwq-32b')

  
# Binding the tools with LLM
tools = [tavily]
llm_with_tools = llm.bind_tools(tools)


graph_builder = StateGraph(State)


# Creating nodes for the graph  
llm_node = LLMNode(llm_with_tools)
tool_node = ToolNode(tools)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)

# Adding edges between nodes
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition)
graph_builder.add_edge("tools", "llm")

agent = graph_builder.compile()


# Generating the response using the agent
if st.button('Generate') and question: 
    messages = agent.invoke({"messages":HumanMessage(content=question)})
    st.write(messages['messages'][3].content)

