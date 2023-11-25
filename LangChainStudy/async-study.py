import asyncio
import time

from langchain.llms import OpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "ChatGLM3-6B-32K")


def invoke_serially():
    for count in range(10):
        resp = llm.invoke("Hello, how are you?")
        print(resp, count)


async def async_invoke(llm):
    resp = await llm.ainvoke("Hello, how are you?")
    print(resp)


async def invoke_concurrently():
    tasks = [async_invoke(llm) for _ in range(10)]
    await asyncio.gather(*tasks)


s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
# await invoke_concurrently()
asyncio.run(invoke_concurrently())
elapsed = time.perf_counter() - s
print("\033[1m" + f"Concurrent executed in {elapsed:0.2f} seconds." + "\033[0m")

s = time.perf_counter()
invoke_serially()
elapsed = time.perf_counter() - s
print("\033[1m" + f"Serial executed in {elapsed:0.2f} seconds." + "\033[0m")