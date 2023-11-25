from langchain.globals import set_llm_cache

from langchain.llms import OpenAI

openai_api_base_address = "http://172.23.115.108:20000/v1"

llm = OpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
        openai_api_base = openai_api_base_address,
        model_name = "ChatGLM3-6B-32K")

""" from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
print(llm.predict("Tell me a joke"))

print("=" * 20)

print(llm.predict("Tell me a joke")) """

# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# The first time, it is not yet in cache, so it should take longer
print(llm.predict("Tell me a joke"))

print("=" * 20)

print(llm.predict("Tell me a joke"))