# from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

full_text = open("./docs/state_of_the_union.txt", "r", encoding='utf-8').read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(full_text)

# embeddings = OpenAIEmbeddings()

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-base")

db = Chroma.from_texts(texts, embeddings)
retriever = db.as_retriever()

retrieved_docs = retriever.invoke(
    "What did the president say about Ketanji Brown Jackson?"
)
print(retrieved_docs[0].page_content)

########################

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


from langchain.chat_models import ChatOpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

chat_model = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "chatglm3-6b-32k")


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
#     | StrOutputParser()
)

print("=" * 80)

""" response = chain.invoke("What did the president say about technology?")
print(response) """

for chunk in chain.stream("What did the president say about technology?"):
    print(chunk.content, end="", flush=True)