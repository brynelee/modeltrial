def fschat_controller_address() -> str:
    from configs.server_config import FSCHAT_CONTROLLER

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = LLM_MODEL) -> str:
    if model := get_model_worker_config(model_name):
        host = model["host"]
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def fschat_openai_api_address() -> str:
    from configs.server_config import FSCHAT_OPENAI_API

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"