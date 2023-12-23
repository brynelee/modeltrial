from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableParallel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_llms import (
    ZhipuAIEmbeddings,
    Zhipuai_LLM,
    load_api
)
api_key = load_api()
model = Zhipuai_LLM(zhipuai_api_key=api_key)

joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
)

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

print(map_chain.invoke({"topic": "bear"}))