from langchain.llms.base import LLM  
from typing import Optional, List, Any, Mapping  
from langchain.callbacks.manager import CallbackManagerForLLMRun  
from http import HTTPStatus  
import dashscope  
from dashscope import Generation  
from langchain.schema import AIMessage, HumanMessage  
import os
import gradio as gr  

""" import sys
import os
module_path = os.path.abspath(os.path.join('..'))
langchainstudy_proj_path = os.path.abspath(os.path.join('../LangChainStudy'))
model_config_path = os.path.abspath(os.path.join('../LangChainStudy/custom_llms'))
sys.path.insert(0, module_path)
sys.path.insert(0, langchainstudy_proj_path)
sys.path.insert(0, model_config_path) """

from LangChainStudy.custom_llms.qwen_llm import DashLLM
  
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

qwllm = DashLLM()  

# 注释1  
def qwen_response(message, history):  
    messages = []  
    for msg in history:  
        messages.append(HumanMessage(content=msg[0]))  
        messages.append(AIMessage(content=msg[1]))  
    messages.append(HumanMessage(content=message))  
    response = qwllm.predict_messages(messages)  
    return response.content  
  
# 注释2  
gr.ChatInterface(qwen_response).launch(server_port=8888)