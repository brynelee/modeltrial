from langchain.llms import OpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "ChatGLM3-6B-32K")

""" for chunk in llm.stream(
    "在通货膨胀和失业之间关系的理论是什么？"):
    print(chunk, end="", flush=True)
 """

response = llm.batch(
    [
        "在通货膨胀和失业之间关系的理论是什么？"
    ]
)

print(response)

