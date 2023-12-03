from modelscope import AutoTokenizer, AutoModelForCausalLM

# Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-72B-Chat-Int4", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/AI_Spaces/models/Qwen-72B-Chat-Int4", trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
#    "qwen/Qwen-72B-Chat-Int4",
    "/mnt/d/AI_Spaces/models/Qwen-72B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# Qwen-72B-Chat现在可以通过调整系统指令（System Prompt），实现角色扮演，语言风格迁移，任务设定，行为设定等能力。
# Qwen-72B-Chat can realize roly playing, language style transfer, task setting, and behavior setting by system prompt.
response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
print(response)
# 哎呀，你好哇！是怎么找到人家的呢？是不是被人家的魅力吸引过来的呀~(≧▽≦)/~

response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
print(response)
# Your colleague is a shining example of dedication and hard work. Their commitment to their job is truly commendable, and it shows in the quality of their work. 
# They are an asset to the team, and their efforts do not go unnoticed. Keep up the great work!