from typing import Annotated,Sequence,TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

try:

    @tool(description="İki adet sayiyi toplamak için var edildi")
    def add(a : int,b : int):
        
        return a + b  # İki tam sayıyı toplayıp sonucu döndüren basit bir fonksiyon
    
    @tool(description="İki adet sayiyi çıkarmak için var edildi")
    def subtract(a : int ,b : int):
        return a - b 
    
    @tool(description="İki adet sayiyi çarpmak için var edildi")
    def multiply(a : int, b:int):
        return a * b 
    

    tools = [add,subtract,multiply]  # 'add' fonksiyonunu içeren araçlar listesi oluşturuluyor

    model = ChatOllama(model="llama3.1:latest").bind_tools(tools)
    # llama3.1:latest modelini çağırıyor ve yukarıdaki araçları (tools) modele bağlıyor

    def model_call(state:AgentState) -> AgentState:
        system_prompt = SystemMessage(content="Sen benim AI asistanımsınm lütfen sorguya cevap ver")
        response = model.invoke([system_prompt] + state["messages"])
        return {"messages" : [response]}


    def should_contiune(state: AgentState) -> AgentState:
        messages = state["messages"]                            # Mevcut mesajları al
        last_message = messages[-1]                             # Son mesajı seç
        if not last_message.tool_calls:                         # Eğer son mesajda tool (araç) çağrısı yoksa
            return "end"                                        # Diyaloğu bitir
        else:
            return "continue"                                   # Araç çağrısı varsa devam et

    graph = StateGraph(AgentState)                              # Bir durum grafiği (state graph) oluştur

    graph.add_node("our_agent", model_call)                     # 'our_agent' adında bir düğüm (node) ekle, model_call fonksiyonunu çalıştıracak

    tool_node = ToolNode(tools=tools)                           # Tanımlı araçları (tools) kullanacak bir ToolNode oluştur
    graph.add_node("tools", tool_node)                          # Bu ToolNode'u 'tools' adıyla grafa ekle

    graph.set_entry_point("our_agent")                          # Diyalogun giriş noktası olarak 'our_agent' düğümünü belirle

    graph.add_conditional_edges(                                # Koşullu geçişler (edge) tanımla
        "our_agent",                                            # Nereden çıkılacak: 'our_agent' düğümünden
        should_contiune,                                        # Hangi koşula göre: should_contiune fonksiyonuna göre
        {"continue": "tools", "end": END}                       # Eğer sonuç 'continue' ise 'tools' düğümüne git, 'end' ise diyalogu bitir
    )

    graph.add_edge("tools","our_agent")

    app = graph.compile()

    def print_stream(stream):
        for s in stream:                                      # Akışın (stream) içindeki her bir öğe (adım) için döngü başlat
            message = s["messages"][-1]                       # Her adımın son mesajını (en güncel mesaj) al
            if isinstance(message, tuple):                    # Eğer bu mesaj bir tuple (demet) ise
                print(message)                                # Olduğu gibi yazdır (tuple'lar özel bir durum olabilir)
            else:
                message.pretty_print()                        # Değilse, mesaj nesnesinin 'pretty_print' metodunu kullanarak daha okunabilir şekilde yazdır


    inputs = {"messages" : [("user", "Add 75 + 61 Subtrach 35 - 21 Add 25 + 31")]}  # Kullanıcıdan gelen mesaj, başlangıç mesajı olarak input'a ekleniyor

    print_stream(app.stream(inputs, stream_mode="values"))  # Agent uygulaması 'stream' modunda çalıştırılıyor ve çıktı akışı yazdırılıyor


except Exception as hata:
    print("Hata : ",hata)

