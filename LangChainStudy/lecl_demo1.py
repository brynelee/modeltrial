from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from custom_llms import (
    ZhipuAIEmbeddings,
    Zhipuai_LLM,
    load_api
)

api_key = load_api()
model = Zhipuai_LLM(zhipuai_api_key=api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n"),
        ("human", "{equation_statement}")
    ]
)

runnable = (
    {"equation_statement": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

print(runnable.invoke("x raised to the third plus seven equals 12"))
