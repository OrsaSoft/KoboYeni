
from typing import TypedDict, List                           # Tip tanımları için: sözlük tipi (TypedDict) ve liste tipi (List)
from langchain_core.messages import HumanMessage            # LangChain'de insan (kullanıcı) mesajlarını temsil eder
from langgraph.graph import StateGraph, START, END          # LangGraph ile akış diyagramı oluşturmak için gerekli sınıflar ve sabitler
from langchain_ollama import ChatOllama                     # Ollama modeliyle yerel dil modeli (LLaMA vs.) entegrasyonu sağlar

class AgentState(TypedDict):                                # Agent'in durumunu temsil eden yapı tanımlanıyor
    messages: List[HumanMessage]                            # Bu durumda sadece insan mesajlarından oluşan bir liste bulunuyor

llm = ChatOllama(model="llama3.1:latest")                   # Yerel çalışacak modeli seçiyoruz (Ollama üzerinden llama3.1)

def process(state: AgentState) -> AgentState:               # Agent'in gerçekleştireceği işlem tanımı
    response = llm.invoke(state["messages"])                # Mesaj listesi modele gönderilir, cevap alınır
    print(f"Yanıt : {response.content}")                    # Modelin cevabı terminale yazdırılır
    return state                                            # Aynı state geri döndürülüyor (tek yönlü, sade akış)

graph = StateGraph(AgentState)                              # LangGraph tabanlı bir akış diyagramı başlatılıyor

graph.add_node("process", process)                          # 'process' adında bir düğüm (node) ekleniyor — model yanıtı burada üretiliyor
graph.add_edge(START, "process")                            # Başlangıçtan process düğümüne bağlantı kuruluyor
graph.add_edge("process", END)                              # process düğümünden sona bağlantı kuruluyor

agent = graph.compile()                                     # Akış diyagramı derlenip çalıştırılabilir hale getiriliyor

user_input = input("Mesajınızı giriniz : ")                 # Kullanıcıdan mesaj alınır

while user_input != "exit":                                 # Kullanıcı 'exit' yazana kadar döngü devam eder
    agent.invoke({"messages": [HumanMessage(content=user_input)]})  # Kullanıcı mesajı modele gönderilir, yanıt alınır
    user_input = input("Mesajınızı giriniz : ")             # Yeni mesaj alınır ve döngü devam eder

