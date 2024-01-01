from langchain.llms.base import LLM  
from typing import Optional, List, Any, Mapping  
from langchain.callbacks.manager import CallbackManagerForLLMRun  
from http import HTTPStatus  
import dashscope  
from dashscope import Generation  
import os
import json  
  
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"] 
  

class DashLLM(LLM):  
    model:str = "qwen-turbo"
    # model:str = "qwen-plus"
    # model:str = "qwen-max"
  
    @property  
    def _llm_type(self) -> str:  
        return "dashllm"  
  
    def _call(  
            self,  
            prompt: str,  
            stop: Optional[List[str]] = None,  
            run_manager: Optional[CallbackManagerForLLMRun] = None,  
            **kwargs: Any,  
    ) -> str:  
        if stop is not None:  
            raise ValueError("stop kwargs are not permitted.")  
        response = Generation.call(  
            model=self.model,  
            prompt=prompt  
        )  
        if response.status_code != HTTPStatus.OK:  
            return f"请求失败，失败信息为:{response.message}"  
        return response.output.text  
  
  
    @property  
    def _identifying_params(self) -> Mapping[str, Any]:  
        """Get the identifying parameters."""  
        return {"model": self.model}  
  
if __name__ == '__main__':  
    qw = DashLLM()  
    print(qw.predict("北京有什么好吃的？"))
