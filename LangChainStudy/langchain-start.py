from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "chatglm3-6b-32k")
chat_model = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "chatglm3-6b-32k")

""" openai_api_base_address = "http://192.168.3.84:20000/v1"
chat_model = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "vicuna-7b") """

from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

print(llm.invoke(text))
# >> Feetful of Fun

print(chat_model.invoke(messages))
# >> AIMessage(content="Socks O'Color")