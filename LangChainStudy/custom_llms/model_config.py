CHAT_CONFIG = {
    'openai': {
        'openai_model': 'gpt-3.5-turbo',
        'openai_api_key': None,  # will use environment  value 'OPENAI_API_KEY' if None
        'llm_kwargs': {
            'temperature': 0.2,
            # 'max_tokens': 200,
            }
    },
    'llama_2': {
        'llama_2_model': 'llama-2-13b-chat',
        'llm_kwargs':{
            'temperature': 0.2,
            'max_tokens': 200,
            'n_ctx': 4096
        }
    },
    'ernie': {
        'ernie_model': 'ernie-bot-turbo',  # 'ernie-bot' or 'ernie-bot-turbo'
        'eb_api_type': None, # If None, use environment  value 'EB_API_TYPE'
        'eb_access_token': None, # If None, use environment value 'EB_ACCESS_TOKEN'
        'llm_kwargs': {}
    },
    'minimax': {
        'minimax_model': 'abab5.5-chat',
        'minimax_api_key': None, # If None, use environment value 'MINIMAX_API_KEY'
        'minimax_group_id': None, # If None, use environment value 'MINIMAX_GROUPID'
        'llm_kwargs': {}
    },
    'dolly': {
        'dolly_model': 'databricks/dolly-v2-3b',
        'llm_kwargs': {'device': 'auto'}
    },
    'skychat': {
        'skychat_api_host': None, # If None, use default value 'sky-api.singularity-ai.com'
        'skychat_app_key': None, # If None, use environment value 'SKYCHAT_APP_KEY'
        'skychat_app_secret': None  # If None, use environment value 'SKYCHAT_APP_SECRET'
    },
    'dashscope': {
        'dashscope_model': 'qwen-plus-v1',
        'dashscope_api_key': None  # If None, use environment value 'DASHSCOPE_API_KEY'
    },
    'chatglm':{
        'chatglm_model': 'chatglm_turbo',
        'chatglm_api_key': None  # If None, use environment value 'ZHIPUAI_API_KEY'
    }
}
