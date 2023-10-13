import asyncio
import sys
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

from server.utils import (fschat_openai_api_address, fschat_controller_address, fschat_model_worker_address, FastAPI, MakeFastAPIOffline)

from typing import List

def create_controller_app(
    dispatch_method: str,
    log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger
    logger.setLevel(log_level)

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller

    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app


def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
    from fastchat.utils import build_logger
    logger = build_logger("openai_api", "openai_api.log")
    logger.setLevel(log_level)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger
    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys

    MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app

def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @app.on_event("startup")
    async def on_startup():
        if started_event is not None:
            started_event.set()


def run_controller(log_level: str = "INFO", started_event: mp.Event = None):

    import uvicorn

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
        log_level=log_level,
    )
    _set_app_event(app, started_event)

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):

    import uvicorn
    from server.utils import set_httpx_config
    set_httpx_config()

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(controller_addr, log_level=log_level)
    _set_app_event(app, started_event)

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    uvicorn.run(app, host=host, port=port)

def run_model_worker():


def run_api_server(started_event: mp.Event = None):
    from server.api import create_app
    import uvicorn
    from server.utils import set_httpx_config
    set_httpx_config()

    


def run_webui():


async def start_main_server():
    
    mp.set_start_method("spawn")
    manager = mp.Manager()

    queue = manager.Queue()

    log_level = "INFO"

    processes = {"online_api": {}, "model_worker": {}}


    ## controller

    controller_started = manager.Event()
    process = Process(
        target=run_controller,
        name=f"controller",
        kwargs=dict(log_level=log_level, started_event=controller_started),
        daemon=True,
    )

    processes["controller"] = process

    ## openai_api

    process = Process(
        target=run_openai_api,
        name=f"openai_api",
        daemon=True,
    )

    processes["openai_api"] = process

    #### model_worker

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

    ## api_server

    api_started = manager.Event()

    process = Process(
        target=run_api_server,
        name=f"API Server",
        kwargs=dict(started_event=api_started),
        daemon=True,
    )
    processes["api"] = process


    ## webui_server

    webui_started = manager.Event()

    process = Process(
        target=run_webui,
        name=f"WEBUI Server",
        kwargs=dict(started_event=webui_started),
        daemon=True,
    )
    processes["webui"] = process 


if __name__ == "__main__":

    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    # 同步调用协程代码
    loop.run_until_complete(start_main_server())
    