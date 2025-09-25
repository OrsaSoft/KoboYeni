# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import create_engine, select,not_
import time as tm
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
import langchain
from langchain_community.embeddings import OllamaEmbeddings
from hashlib import sha256
import os




print(f"LangChain version: {langchain.__version__}") # 0.3.27

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db_path = "./vectordb"

vector_db = Chroma(persist_directory=db_path,embedding_function=embeddings)



# 6️⃣ Retriever ve LLM kısmı
retriever = vector_db.as_retriever(search_kwargs={"k" : 100})
llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir yapay zeka asistanısın. Bu Belgeler hakkında sana soru sorulacak {context}"),
    ("human", "{input}"),
])

combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rga_chain = create_retrieval_chain(retriever, combine_chain)
print("işlem bitti")



while True:
    mesaj = input("Soru sormak için buraya yazın (çıkmak için 1): ")
    
    if mesaj == "1":
        print("Programdan çıkılıyor...")
        break
    
    
    docs = retriever.get_relevant_documents(mesaj) # Kullanıcının sorusuna en yakın belgeleri getirir isterse uzunluğu aluınabilir 
    print(f"Retrieved {len(docs)} documents")

    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print("Content:", doc.page_content[:500])  # İlk 500 karakterini göster
        print("Metadata:", doc.metadata)

    # vectordb\aa1bf15c-2132-4fc5-8849-730b2e89bbe6\data_level0.bin
    # vectordb\sqlite3

    # Sorguyu çalıştır
    response = rga_chain.invoke({"input": mesaj})
    
    # Süreyi bitir
    end = tm.time()

    
    print("AI'nın Cevabı Gemma D:", response["answer"])

    # What does the US government expect of Bytedance ?


