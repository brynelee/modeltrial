import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
# from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs import LLM_MODEL, TEMPERATURE, HISTORY_LEN
from server.utils import get_model_worker_config
from typing import List, Dict
import json

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)


    
def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:

    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)



def dialogue_page(api: ApiRequest):
    chat_box.init_session()

    with st.sidebar:

        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)

        dialogue_mode = st.selectbox("请选择对话模式：",
                                     ["LLM 对话",
                                      "知识库问答",
                                      "搜索引擎问答",
                                      "自定义Agent问答",
                                      ],
                                     index=0,
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )
        
        llm_model = "chatglm2-6b-32k"

        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.01)

        ## 部分模型可以超过10抡对话
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)


    # Display chat messages from history on app rerun

    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM 对话":
            chat_box.ai_say("正在思考...")
            text = ""
            r = api.chat_chat(prompt, history=history, model=llm_model, temperature=temperature)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标


    now = datetime.now()
    with st.sidebar:

        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.experimental_rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )