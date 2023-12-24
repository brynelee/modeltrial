import requests
import readline

# 这个是官方API说明，直接使用url访问方式，比较复杂，可参考custom_llms包对其的封装使用

import os
group_id = os.environ["MINIMAX_GROUPID"]
api_key = os.environ["MINIMAX_API_KEY"]

url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={group_id}"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# tokens_to_generate/bot_setting/reply_constraints可自行修改
request_body = payload = {
    "model": "abab5.5-chat",
    "tokens_to_generate": 1024,
    "reply_constraints": {"sender_type": "BOT", "sender_name": "MM智能助理"},
    "messages": [],
    "bot_setting": [
        {
            "bot_name": "MM智能助理",
            "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。",
        }
    ],
}
# 添加循环完成多轮交互
while True:
    # 下面的输入获取是基于python终端环境，请根据您的场景替换成对应的用户输入获取代码
    line = input("发言:")
    # 将当次输入内容作为用户的一轮对话添加到messages
    request_body["messages"].append(
        {"sender_type": "USER", "sender_name": "小明", "text": line}
    )
    response = requests.post(url, headers=headers, json=request_body)
    reply = response.json()["reply"]
    print(f"reply: {reply}")
    #  将当次的ai回复内容加入messages
    request_body["messages"].extend(response.json()["choices"][0]["messages"])