import asyncio
import time

from langchain.llms import OpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "ChatGLM3-6B-32K")

s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
# await llm.abatch(["Hello, how are you?"] * 10)
asyncio.run(llm.abatch(["Hello, how are you?"] * 10))
elapsed = time.perf_counter() - s
print("\033[1m" + f"Batch executed in {elapsed:0.2f} seconds." + "\033[0m")