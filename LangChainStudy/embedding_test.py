from langchain.embeddings import HuggingFaceEmbeddings

# embedding_model = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-base")

# embedding_model = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-large")



embedding_model = HuggingFaceEmbeddings(model_name = "shibing624/text2vec-base-chinese")

print("ok")