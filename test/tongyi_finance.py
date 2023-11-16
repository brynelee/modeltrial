from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = snapshot_download('TongyiFinance/Tongyi-Finance-14B-Chat')

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
# 模型加载指定device_map='cuda:0'，更改成device_map='auto'会使用所有可用显卡

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

response, history = model.chat(tokenizer, "请解释一下资产负债率", history=None)
print(response)
# 资产负债率是一个财务比率，用来衡量一个企业的负债水平。它是用一个企业负债总额除以其资产总额的百分比来表示的。它的计算公式是：资产负债率 = 负债总额 / 资产总额。它能够反映一个企业的财务状况，以及它是否具有足够的资产来抵偿其债务。