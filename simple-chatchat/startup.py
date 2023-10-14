import asyncio
import sys
import os
import subprocess
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

from server.utils import (fschat_openai_api_address, fschat_controller_address, fschat_model_worker_address, FastAPI, MakeFastAPIOffline, get_model_worker_config)

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


def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    kwargs包含的字段如下：
    host:
    port:
    model_names:[`model_name`]
    controller_address:
    worker_address:


    对于online_api:
        online_api:True
        worker_class: `provider`                                            
    对于离线模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    """
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.model_worker import worker_id, logger
    import argparse
    logger.setLevel(log_level)

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)

    from configs.model_config import VLLM_MODEL_DICT
    
    if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
        
        print("=" * 30 + "run into vllm part" + "=" * 30)

    else:
        from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker
        args.gpus = "0"
        args.max_gpu_memory = "20GiB"
        args.num_gpus = 1

        args.log_8bit=False
        args.cpu_offloading = None
        args.gptq_ckpt = None
        args.gptq_wbits = 16
        args.gptq_groupsize = -1
        args.gptq_act_order = False
        args.awq_ckpt = None
        args.awq_wbits = 16
        args.awq_groupsize = -1
        args.model_names = []
        args.conv_template = None
        args.limit_worker_concurrency = 5
        args.stream_interval = 2
        args.no_register = False
        args.embed_in_truncate = False
        for k, v in kwargs.items():
            setattr(args, k, v)
        if args.gpus:
            if args.num_gpus is None:
                args.num_gpus = len(args.gpus.split(','))
            if len(args.gpus.split(",")) < args.num_gpus:
                raise ValueError(
                    f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        gptq_config = GptqConfig(
            ckpt=args.gptq_ckpt or args.model_path,
            wbits=args.gptq_wbits,
            groupsize=args.gptq_groupsize,
            act_order=args.gptq_act_order,
        )
        awq_config = AWQConfig(
            ckpt=args.awq_ckpt or args.model_path,
            wbits=args.awq_wbits,
            groupsize=args.awq_groupsize,
        )

        worker = ModelWorker(
            controller_addr=args.controller_address,
            worker_addr=args.worker_address,
            worker_id=worker_id,
            model_path=args.model_path,
            model_names=args.model_names,
            limit_worker_concurrency=args.limit_worker_concurrency,
            no_register=args.no_register,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            stream_interval=args.stream_interval,
            conv_template=args.conv_template,
            embed_in_truncate=args.embed_in_truncate,
        )
        sys.modules["fastchat.serve.model_worker"].args = args
        sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
        sys.modules["fastchat.serve.model_worker"].worker = worker

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
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

def run_model_worker(
        model_name: str = LLM_MODEL,
        controller_address: str = "",
        log_level: str = "INFO",
        q: mp.Queue = None,
        started_event: mp.Event = None,
):
    import uvicorn
    from server.utils import set_httpx_config
    set_httpx_config()

    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    app = create_model_worker_app(log_level=log_level, **kwargs)
    _set_app_event(app, started_event)

    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())



def run_api_server(started_event: mp.Event = None):
    from server.api import create_app
    import uvicorn
    from server.utils import set_httpx_config
    set_httpx_config()

    app = create_app()
    _set_app_event(app, started_event)

    host = API_SERVER("host")
    port = API_SERVER("port")

    uvicorn.run(app, host=host, port=port)


def run_webui(started_event: mp.Event = None):
    from server.utils import set_httpx_config
    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    p = subprocess.Popen(["streamlit", "run", "webui.py",
                          "--server.address", host,
                          "--server.port", str(port),
                          "--theme.base", "light",
                          "--theme.primaryColor", "#165dff",
                          "--theme.secondaryBackgroupdColor", "Ef5f5f5",
                          "--theme.textColor", "#000000",
                          ])
    started_event.set()
    p.wait()


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
    