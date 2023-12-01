from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate user input into pirate speak",
        ),
        ("human", "{text}"),
    ]
)
# _model = ChatOpenAI()

openai_api_base_address = "http://192.168.3.84:20000/v1"

_model = ChatOpenAI(openai_api_key = "aaabbbcccdddeeefffedddsfasdfasdf", 
    openai_api_base = openai_api_base_address,
    model_name = "chatglm3-6b-32k")

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
