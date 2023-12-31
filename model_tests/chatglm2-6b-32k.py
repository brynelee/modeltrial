
from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained("/mnt/d/AI_Spaces/models/chatglm3-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/d/AI_Spaces/models/chatglm3-6b-32k", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "请给出快速排序算法的python示例", history=history)
print(response)