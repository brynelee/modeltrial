# 实际使用
# xdmortar ubuntu
# export VLLM_USE_MODELSCOPE=True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python -m vllm.entrypoints.openai.api_server \
    --model="/home/xiaodong/Gitroot/tools/model/Yi-34B-Chat-4bits" \
    --served-model-name 01ai/Yi-34B-Chat-4bits \
    --trust-remote-code \
    --max-model-len 2048 -q awq \
    --gpu-memory-utilization 1 \
    --enforce-eager

