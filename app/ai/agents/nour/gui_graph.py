from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.schemas.blocks import MessageWithBlocks
import os

class NourState(TypedDict):
    messages: list[BaseMessage]
    final_blocks: MessageWithBlocks | None

def call_llm(state: NourState):
    llm = ChatOpenAI(model='gpt-4o-mini', base_url='https://openrouter.ai/api/v1/chat/completions', api_key=os.environ.get('OPENROUTER_API_KEY'))
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are Nour, a researcher. You MUST output a highly structured JSON array of UI blocks. Use TextBlock for chat, FindingsBlock for insights.'),
        ('placeholder', '{messages}')
    ])
    structured_llm = llm.with_structured_output(MessageWithBlocks)
    chain = prompt | structured_llm
    response = chain.invoke({'messages': state['messages']})
    return {'final_blocks': response}

graph_builder = StateGraph(NourState)
graph_builder.add_node('llm', call_llm)
graph_builder.add_edge(START, 'llm')
graph_builder.add_edge('llm', END)
graph = graph_builder.compile()