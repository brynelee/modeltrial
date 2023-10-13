
from configs import VERSION
from configs.server_config import OPEN_CROSS_DOMAIN

from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from server.utils import BaseResponse, FastAPI, MakeFastAPIOffline

async def document():
    return RedirectResponse(url="/docs")

def create_app():
    app = FastAPI(
        title = "Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)

    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)
    


