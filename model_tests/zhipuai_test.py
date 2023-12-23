import zhipuai
import os
from dotenv import find_dotenv, load_dotenv
def load_api():
    try:
        _ = load_dotenv(find_dotenv())
        api_key = os.environ["ZHIPUAI_API_KEY"]    #填写控制台中获取的 APIKey 信息
    except Exception as e:
        print(e)
        ValueError("Please input correct ZHIPUAI_API_KEY!")
    return api_key

zhipuai.api_key = load_api()
response = zhipuai.model_api.invoke(
    model="chatglm_turbo",
    prompt=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我是人工智能助手"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "assistant", "content": "我叫chatGLM"},
        {"role": "user", "content": "你都可以做些什么事"},
    ]
)

print(response)
print("=" * 50)
print(response["data"]["choices"][-1]["content"].strip('"').strip(" "))