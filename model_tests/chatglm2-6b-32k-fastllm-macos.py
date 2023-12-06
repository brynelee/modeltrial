
from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained("/Users/xiaodong/GitRoot/tools/models/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/Users/xiaodong/GitRoot/tools/models/chatglm2-6b", trust_remote_code=True)

# 加入下面这两行，将huggingface模型转换成fastllm模型
# 目前from_hf接口只能接受原始模型，或者ChatGLM的int4, int8量化模型，暂时不能转换其它量化模型
from fastllm_pytools import llm
model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"

# model = model.eval()

""" response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "请给出快速排序算法的python示例", history=history)
print(response) """

query = "请给出快速排序算法的python示例"
history = []
curResponse = ""
for response in model.stream_response(query, history = history):
    curResponse += curResponse
    print(response, flush = True, end= "")
    history.append((query, curResponse))
