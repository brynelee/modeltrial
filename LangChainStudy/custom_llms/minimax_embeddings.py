import os

group_id = os.environ["MINIMAX_GROUPID"]
api_key = os.environ["MINIMAX_API_KEY"]
from langchain.embeddings import MiniMaxEmbeddings
embeddings = MiniMaxEmbeddings(minimax_group_id=group_id, minimax_api_key=api_key)

query_text = "这是一个测试查询。"
query_result = embeddings.embed_query(query_text)
document_text = "这是一个测试文档。"
document_result = embeddings.embed_documents([document_text])

import numpy as np
query_numpy = np.array(query_result)
document_numpy = np.array(document_result[0])
similarity = np.dot(query_numpy, document_numpy) / (np.linalg.norm(query_numpy) * np.linalg.norm(document_numpy))
print(f"文档与查询之间的余弦相似度：{similarity}")
