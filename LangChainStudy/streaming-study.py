from langchain.chat_models import ChatOpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "ChatGLM3-6B-32K")

for chunk in llm.stream("Write me a song about goldfish on the moon"):
    print(chunk.content, end="", flush=True)