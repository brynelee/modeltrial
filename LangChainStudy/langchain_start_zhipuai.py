from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from custom_llms import (load_api, Zhipuai_LLM)

api_key = load_api()
# print(api_key)
llm = Zhipuai_LLM(zhipuai_api_key=api_key)
# print(llm("给我讲段笑话"), sep="\n") 

from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

print(llm.invoke(text))
# >> Feetful of Fun

print(llm.invoke(messages))
# >> AIMessage(content="Socks O'Color")