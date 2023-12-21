from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from custom_llms import (
    ZhipuAIEmbeddings,
    Zhipuai_LLM,
    load_api
)

api_key = load_api()
model = Zhipuai_LLM(zhipuai_api_key=api_key)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=ZhipuAIEmbeddings()
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(retrieval_chain.invoke("where did harrison work?"))