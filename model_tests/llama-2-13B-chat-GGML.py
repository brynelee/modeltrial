
# reference https://zhuanlan.zhihu.com/p/649366179

from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(openai_api_key = "EMPTY", openai_api_base = "http://localhost:20000/v1", max_tokens=256)

from langchain.schema import AIMessage, HumanMessage, SystemMessage

system_text = "You are a helpful assistant."
human_text1 = "What is the capital of France?"
assistant_text = "Paris."
human_text2 = "How about England?"

messages = [SystemMessage(content=system_text), 
            HumanMessage(content=human_text1), 
            AIMessage(content=assistant_text), 
            HumanMessage(content=human_text2)]

chat_model.predict_messages(messages)

# ================

from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("./README.md")
text = loader.load()

# 或者
# from langchain.document_loaders import TextLoader
# loader = TextLoader("README.md")
# loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 400,
    length_function = len,
    is_separator_regex = False
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
vectorstore = FAISS.from_documents(documents=text_splitter.split_documents(text), embedding=HuggingFaceEmbeddings(model_name = "BAAI/bge-large-zh"))



import requests
from langchain.embeddings.base import Embeddings

class LocalLlamaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        url = "http://localhost:20000/v1/embeddings"
        request_body = {
            "input": texts
        }
        response = requests.post(url, json=request_body)
        return [data["embedding"] for data in response.json()["data"]]

    def embed_query(self, text):
        url = "http://localhost:20000/v1/embeddings"
        request_body = {
            "input": text
        }
        response = requests.post(url, json=request_body)
        return response.json()["data"][0]["embedding"]
    
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents=all_splits, embedding=LocalLlamaEmbeddings())

question = "How to run the program in interactive mode?"
docs = vectorstore.similarity_search(question, k=1)

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

chat_model = ChatOpenAI(openai_api_key = "EMPTY", openai_api_base = "http://localhost:8000/v1", max_tokens=256)

qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=vectorstore.as_retriever(search_kwargs={"k": 1}))
qa_chain({"query": "How to run the program in interactive mode?"})


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    chat_model,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
qa_chain({"query": "What is --interactive option used for?"})

