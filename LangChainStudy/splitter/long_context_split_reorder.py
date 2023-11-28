from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

# Get embeddings.
# embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-large")
embeddings = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-base")

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my favourite songs.",
    "The Celtics are my favourite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larry Bird was an iconic NBA player.",
]

# Create a retriever
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
query = "What can you tell me about the Celtics?"

# Get relevant documents ordered by relevance score
docs = retriever.get_relevant_documents(query)
print(docs)

print("=" * 80)
print("=" * 80)

# Reorder the documents:
# Less relevant document will be at the middle of the list and more
# relevant elements at beginning / end.
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# Confirm that the 4 relevant documents are at beginning and end.
print(reordered_docs)

print("=" * 80)
print("=" * 80)


# We prepare and run a custom Stuff chain with reordered docs as context.

from langchain.llms import OpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "vicuna-13b-v1.5")

# Override prompts
document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"

stuff_prompt_override = """Given this text extracts:
-----
{context}
-----
Please answer the following question:
{query}"""
prompt = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)

# Instantiate the chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)
response = chain.run(input_documents=reordered_docs, query=query)
print(response)

print("=" * 80)
