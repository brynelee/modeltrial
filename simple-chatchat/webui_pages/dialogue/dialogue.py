import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
# from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs import LLM_MODEL, TEMPERATURE
from server.utils import get_model_worker_config
from typing import List, Dict

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)

def dialogue_page(api: ApiRequest):
    now = datetime.now()