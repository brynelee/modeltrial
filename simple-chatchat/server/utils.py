from fastapi import FastAPI
from pathlib import Path

import pydantic
from pydantic import BaseModel

from configs import (HTTPX_DEFAULT_TIMEOUT, MODEL_PATH, MODEL_ROOT_PATH)

from typing import Union, Dict, Any


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    if type in MODEL_PATH:
        paths = MODEL_PATH[type]
    else:
        paths = {}
        for v in MODEL_PATH.values():
            paths.update(v)

    if path_str := paths.get(model_name):  # 以 "chatglm-6b": "THUDM/chatglm-6b-new" 为例，以下都是支持的路径
        path = Path(path_str)
        if path.is_dir():  # 任意绝对路径
            return str(path)

        root_path = Path(MODEL_ROOT_PATH)
        if root_path.is_dir():
            path = root_path / model_name
            if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
                return str(path)
            path = root_path / path_str
            if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
                return str(path)
            path = root_path / path_str.split("/")[-1]
            if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
                return str(path)
        return path_str  # THUDM/chatglm06b

# 从server_config中获取服务信息
def get_model_worker_config(model_name: str = None) -> dict:
    '''
    加载model worker的配置项。
    优先级:FSCHAT_MODEL_WORKERS[model_name] > ONLINE_LLM_MODEL[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    from configs.model_config import ONLINE_LLM_MODEL
    from configs.server_config import FSCHAT_MODEL_WORKERS
    from server import model_workers

    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()

    config["model_path"] = get_model_path(model_name)
    config["device"] = llm_device(config.get("device"))
    return config

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


def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(i)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )
        

def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
):
    '''
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    '''
    import httpx
    import os

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    # TODO: 简单的清除系统代理不是个好的选择，影响太多。似乎修改代理服务器的bypass列表更好。
    # patch requests to use custom proxies instead of system settings
    # def _get_proxies():
    #     return {}

    # import urllib.request
    # urllib.request.getproxies = _get_proxies