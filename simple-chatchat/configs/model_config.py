




# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录
MODEL_ROOT_PATH = ""

# 在以下字典中修改属性值，以指定本地embedding模型存储位置。支持3种设置方法：
# 1、将对应的值修改为模型绝对路径
# 2、不修改此处的值（以 text2vec 为例）：
#       2.1 如果{MODEL_ROOT_PATH}下存在如下任一子目录：
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 如果以上本地路径不存在，则使用huggingface模型
MODEL_PATH = {
    "embed_model": {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "GanymedeNil/text2vec-large-chinese",
        "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
        "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
        "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
        "text2vec-bge-large-chinese": "shibing624/text2vec-bge-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
        "m3e-large": "moka-ai/m3e-large",
        "bge-small-zh": "BAAI/bge-small-zh",
        "bge-base-zh": "BAAI/bge-base-zh",
        "bge-large-zh": "BAAI/bge-large-zh",
        "bge-large-zh-noinstruct": "BAAI/bge-large-zh-noinstruct",
        "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
        "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
        "piccolo-base-zh": "sensenova/piccolo-base-zh",
        "piccolo-large-zh": "sensenova/piccolo-large-zh",
        "text-embedding-ada-002": "your OPENAI_API_KEY",
    },
    # TODO: add all supported llm models
    "llm_model": {
        # 以下部分模型并未完全测试，仅根据fastchat和vllm模型的模型列表推定支持
        "chatglm-6b": "THUDM/chatglm-6b",
        # "chatglm2-6b": "THUDM/chatglm2-6b",
        "chatglm2-6b": "D:\\AI_Spaces\\ChatGLM2\\text-generation-webui\\models\\chatglm2-6b",
        "chatglm2-6b-int4": "THUDM/chatglm2-6b-int4",
        # "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
        "ChatGLM3-6B-32K": "/mnt/d/AI_Spaces/models/chatglm3-6b-32k",

        "baichuan2-13b": "baichuan-inc/Baichuan-13B-Chat",
        "baichuan2-7b":"baichuan-inc/Baichuan2-7B-Chat",

        "baichuan-7b": "baichuan-inc/Baichuan-7B",
        "baichuan-13b": "baichuan-inc/Baichuan-13B",
        'baichuan-13b-chat':'baichuan-inc/Baichuan-13B-Chat',

        "aquila-7b":"BAAI/Aquila-7B",
        "aquilachat-7b":"BAAI/AquilaChat-7B",

        "internlm-7b":"internlm/internlm-7b",
        "internlm-chat-7b":"internlm/internlm-chat-7b",

        "falcon-7b":"tiiuae/falcon-7b",
        "falcon-40b":"tiiuae/falcon-40b",
        "falcon-rw-7b":"tiiuae/falcon-rw-7b",

        "gpt2":"gpt2",
        "gpt2-xl":"gpt2-xl",
        
        "gpt-j-6b":"EleutherAI/gpt-j-6b",
        "gpt4all-j":"nomic-ai/gpt4all-j",
        "gpt-neox-20b":"EleutherAI/gpt-neox-20b",
        "pythia-12b":"EleutherAI/pythia-12b",
        "oasst-sft-4-pythia-12b-epoch-3.5":"OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "dolly-v2-12b":"databricks/dolly-v2-12b",    
        "stablelm-tuned-alpha-7b":"stabilityai/stablelm-tuned-alpha-7b",

        "Llama-2-13b-hf":"meta-llama/Llama-2-13b-hf",    
        "Llama-2-70b-hf":"meta-llama/Llama-2-70b-hf",
        "open_llama_13b":"openlm-research/open_llama_13b",    
        # "vicuna-13b-v1.3":"lmsys/vicuna-13b-v1.3",
        "vicuna-13b-v1.3":"D:\\AI_Spaces\\models\\vicuna-13b-v1.3",
        "koala":"young-geng/koala",  
          
        "mpt-7b":"mosaicml/mpt-7b",
        "mpt-7b-storywriter":"mosaicml/mpt-7b-storywriter",    
        "mpt-30b":"mosaicml/mpt-30b",
        "opt-66b":"facebook/opt-66b",    
        "opt-iml-max-30b":"facebook/opt-iml-max-30b",

        "Qwen-7B":"Qwen/Qwen-7B",
        "Qwen-14B":"Qwen/Qwen-14B",
        # "Qwen-7B-Chat":"Qwen/Qwen-7B-Chat",
        "Qwen-7B-Chat":"C:\\Users\\bryne\\.cache\\modelscope\\hub\\qwen\\Qwen-7B-Chat",
        # "Qwen-14B-Chat":"Qwen/Qwen-14B-Chat",
        "Qwen-14B-Chat":"D:\AI_Spaces\models\Qwen-14B-Chat",

        "WizardCoder-Python-13B-V1.0": "D:\\AI_Spaces\\ChatGLM2\\text-generation-webui\\models\\WizardCoder-Python-13B-V1.0",
                
    },
}

# 选用的 Embedding 名称
EMBEDDING_MODEL = "m3e-base" # 可以尝试最新的嵌入式sota模型：piccolo-large-zh


# Embedding 模型运行设备。设为"auto"会自动检测，也可手动设定为"cuda","mps","cpu"其中之一。
EMBEDDING_DEVICE = "auto"

# LLM 名称
# LLM_MODEL = "chatglm2-6b"
# LLM_MODEL = "chatglm2-6b-32k"
# LLM_MODEL = "Qwen-14B-Chat"
# LLM_MODEL = "WizardCoder-Python-13B-V1.0"
# LLM_MODEL = "vicuna-13b-v1.3"
# LLM_MODEL = "Qwen-7B-Chat"
LLM_MODEL = "ChatGLM3-6B-32K"

# LLM 运行设备。设为"auto"会自动检测，也可手动设定为"cuda","mps","cpu"其中之一。
LLM_DEVICE = "auto"

# 历史对话轮数
HISTORY_LEN = 3

# LLM通用对话参数
TEMPERATURE = 0.7
# TOP_P = 0.95 # ChatOpenAI暂不支持该参数

VLLM_MODEL_DICT = {
    "aquila-7b":"BAAI/Aquila-7B",
    "aquilachat-7b":"BAAI/AquilaChat-7B",

    "baichuan-7b": "baichuan-inc/Baichuan-7B",
    "baichuan-13b": "baichuan-inc/Baichuan-13B",
    'baichuan-13b-chat':'baichuan-inc/Baichuan-13B-Chat',
    # 注意：bloom系列的tokenizer与model是分离的，因此虽然vllm支持，但与fschat框架不兼容
    # "bloom":"bigscience/bloom",
    # "bloomz":"bigscience/bloomz",
    # "bloomz-560m":"bigscience/bloomz-560m",
    # "bloomz-7b1":"bigscience/bloomz-7b1",
    # "bloomz-1b7":"bigscience/bloomz-1b7",

    "internlm-7b":"internlm/internlm-7b",
    "internlm-chat-7b":"internlm/internlm-chat-7b",
    "falcon-7b":"tiiuae/falcon-7b",
    "falcon-40b":"tiiuae/falcon-40b",
    "falcon-rw-7b":"tiiuae/falcon-rw-7b",
    "gpt2":"gpt2",
    "gpt2-xl":"gpt2-xl",
    "gpt-j-6b":"EleutherAI/gpt-j-6b",
    "gpt4all-j":"nomic-ai/gpt4all-j",
    "gpt-neox-20b":"EleutherAI/gpt-neox-20b",
    "pythia-12b":"EleutherAI/pythia-12b",
    "oasst-sft-4-pythia-12b-epoch-3.5":"OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "dolly-v2-12b":"databricks/dolly-v2-12b",    
    "stablelm-tuned-alpha-7b":"stabilityai/stablelm-tuned-alpha-7b",
    "Llama-2-13b-hf":"meta-llama/Llama-2-13b-hf",    
    "Llama-2-70b-hf":"meta-llama/Llama-2-70b-hf",
    "open_llama_13b":"openlm-research/open_llama_13b",    
    "vicuna-13b-v1.3":"lmsys/vicuna-13b-v1.3",
    "koala":"young-geng/koala",    
    "mpt-7b":"mosaicml/mpt-7b",
    "mpt-7b-storywriter":"mosaicml/mpt-7b-storywriter",    
    "mpt-30b":"mosaicml/mpt-30b",
    "opt-66b":"facebook/opt-66b",    
    "opt-iml-max-30b":"facebook/opt-iml-max-30b",

    "Qwen-7B":"Qwen/Qwen-7B",
    "Qwen-14B":"Qwen/Qwen-14B",
    "Qwen-7B-Chat":"Qwen/Qwen-7B-Chat",
    "Qwen-14B-Chat":"Qwen/Qwen-14B-Chat",

}