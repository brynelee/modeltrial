
import multiprocessing as mp
from multiprocessing import Process

from configs import (
    LOG_PATH,
    log_verbose,
    logger,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TEXT_SPLITTER_NAME,
    FSCHAT_CONTROLLER,
    FSCHAT_OPENAI_API,
    API_SERVER,
    WEBUI_SERVER,
    HTTPX_DEFAULT_TIMEOUT,
)

from server.utils import (fschat_openai_api_address, fschat_controller_address, fschat_model_worker_address)

def run_controller():


def run_openai_api():


def run_model_worker():


async def start_main_server():
    
    mp.set_start_method("spawn")
    manager = mp.Manager()

    queue = manager.Queue()

    log_level = "INFO"

    processes = {"online_api": {}, "model_worker": {}}

    controller_started = manager.Event()
    process = Process(
        target=run_controller,
        name=f"controller",
        kwargs=dict(log_level=log_level, started_event=controller_started),
        daemon=True,
    )

    processes["controller"] = process

    process = Process(
        target=run_openai_api,
        name=f"openai_api",
        daemon=True,
    )

    processes["openai_api"] = process

    model_worker_started = []

    e = manager.Event()
    model_worker_started.append(e)

    model_name = "chatglm2-6b-32k"

    controller_address = fschat_controller_address()

    process = Process(
        target=run_model_worker,
        name=f"model_worker - {model_name}",
        kwargs=dict(model_name=model_name,
                    controller_address=controller_address,
                    log_level=log_level,
                    q=queue,
                    started_event = e),
        daemon=True,
    )

    processes["model_worker"][model_name] = process



    