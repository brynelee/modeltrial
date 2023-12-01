#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
from pirate_speak.chain import chain as pirate_speak_chain

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(app, pirate_speak_chain, path="/pirate-speak")

openai_api_base_address = "http://192.168.3.84:20000/v1"

chat_model = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "chatglm3-6b-32k")

""" add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

add_routes(
    app,
    ChatAnthropic(),
    path="/anthropic",
) """

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

add_routes(
    app,
    prompt | chat_model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="192.168.3.84", port=8000)