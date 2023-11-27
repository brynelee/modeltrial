from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./LangChainStudy/demo_text.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# embedding_model = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-base")

from langchain.embeddings import LocalAIEmbeddings

openai_api_base_address = "http://172.23.115.108:20000/v1"

# 这个可以工作，使用服务化embedding模型
embedding_model=LocalAIEmbeddings(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf",  
                            openai_api_base = openai_api_base_address,
                            model = "vicuna-13b-v1.5")

db = Chroma.from_documents(documents, embedding_model)

query = "how to customize your text splitter"
docs = db.similarity_search(query)
print(docs[0].page_content)

print("=" * 80)
print("=" * 80)

embedding_vector = embedding_model.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)