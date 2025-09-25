import os
from typing import TypedDict, List,Union                           # Tip tanımları için: sözlük tipi (TypedDict) ve liste tipi (List)
from langchain_core.messages import HumanMessage,AIMessage            # LangChain'de insan (kullanıcı) mesajlarını temsil eder
from langgraph.graph import StateGraph, START, END          # LangGraph ile akış diyagramı oluşturmak için gerekli sınıflar ve sabitler
from langchain_ollama import ChatOllama                     # Ollama modeliyle yerel dil modeli (LLaMA vs.) entegrasyonu sağlar

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]] # Hem İnsan Mesajları hem AI mesajlarını içerir 

llm = ChatOllama(model="llama3.1:latest")

def process(state :AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print("AI  : ", response.content)
    # İşte tam burada AI cevabı, mesajlar listesine (state["messages"]) ekleniyor!
    state["messages"].append(AIMessage(content=response.content)) # AI'dan alınan cevap mesaj listesine ekleniyor 

    print("Current State : ",state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Bir mesaj giriniz : ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input)) # insan mesajları modele gönderilir 
    result = agent.invoke({"messages" : conversation_history}) # Modelden gelen cevap alınır şöyle olur "messages": [HumanMessage(...), AIMessage(...)]

    conversation_history = result["messages"]
    user_input = input("Bir mesaj giriniz : ")
    

