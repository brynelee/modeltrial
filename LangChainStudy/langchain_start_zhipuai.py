from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from llms import (load_api, Zhipuai_LLM)

""" openai_api_base_address = "http://127.0.0.1:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "zhipu-api") """
""" chat_model = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "chatglm3-6b-32k") """

api_key = load_api()
# print(api_key)
llm = Zhipuai_LLM(zhipuai_api_key=api_key)
print(llm("给我讲段笑话"), sep="\n") 

""" from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

print(llm.invoke(text))
# >> Feetful of Fun

print(chat_model.invoke(messages))
# >> AIMessage(content="Socks O'Color") """