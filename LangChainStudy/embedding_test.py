from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-base")

# print("ok")

# embedding_model = HuggingFaceEmbeddings(model_name = "moka-ai/m3e-large")

print("ok")

# embedding_model = HuggingFaceEmbeddings(model_name = "shibing624/text2vec-base-chinese")

print("ok")

""" from langchain.embeddings import LocalAIEmbeddings

openai_api_base_address = "http://172.23.115.108:20000/v1"

# 这个可以工作，使用服务化embedding模型
embedding_model=LocalAIEmbeddings(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf",  
                            openai_api_base = openai_api_base_address,
                            model = "chatglm3-6b-32k") """

