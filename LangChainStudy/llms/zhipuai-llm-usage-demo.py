import zhipuai

zhipuai.api_key = "0de81c20cb9e8f5677b5282963404b5f.dTwvxLRmTZqPz2Jv"
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

print(response['data']['choices'])