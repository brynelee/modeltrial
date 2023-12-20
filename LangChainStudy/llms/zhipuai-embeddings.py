from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    zhipuai_api_key: Optional[str] = None
    """Zhipuai application apikey"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        验证环境变量或配置文件中的zhipuai_api_key是否可用。

        Args:

            values (Dict): 包含配置信息的字典，必须包含 zhipuai_api_key 的字段
        Returns:

            values (Dict): 包含配置信息的字典。如果环境变量或配置文件中未提供 zhipuai_api_key，则将返回原始值；否则将返回包含 zhipuai_api_key 的值。
        Raises:

            ValueError: zhipuai package not found, please install it with `pip install
            zhipuai`
        """
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            import zhipuai
            zhipuai.api_key = values["zhipuai_api_key"]
            values["client"] = zhipuai.model_api

        except ImportError:
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values

    def _embed(self, texts: str) -> List[float]:
        """
        生成输入文本的 embedding。
        
        Args:
            texts (str): 要生成 embedding 的文本。

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表。
        """
        try:
            resp = self.client.invoke(
                model="text_embedding",
                prompt=texts
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if resp["code"] != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (resp["code"], resp["msg"])
            )
        embeddings = resp["data"]["embedding"]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding。
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding。
        
        Args:
            text (str): 要生成 embedding 的文本。

        Return:
            List [float]: 输入文本的 embedding，一个浮点数值列表。
        """
        resp = self.embed_documents([text])
        return resp[0]





from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./LangChainStudy/demo_text.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# 这个可以工作，使用服务化embedding模型
embedding_model=ZhipuAIEmbeddings()

db = Chroma.from_documents(documents, embedding_model)

query = "how to customize your text splitter"
docs = db.similarity_search(query)
print(docs[0].page_content)

print("=" * 80)
print("=" * 80)

embedding_vector = embedding_model.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)