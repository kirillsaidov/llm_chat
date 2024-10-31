# module chatui

# system
import os
import re
import sys
import time
import argparse
import requests

# webui
import streamlit as st


def ollama_chat(messages: list[dict], model: str, url: str = 'localhost', stream: bool = False, keep_alive: bool = True) -> requests.Response:
    """Generate ollama response

    Args:
        messages (list[dict]): list of messages
        model (str, optional): model identifier
        url (str, optional): url to ollama server. Defaults to 'localhost'.
        stream (bool, optional): stream response. Defaults to False.
        keep_alive (bool, optional): keep model loaded in memory indefinitely. Defaults to True.

    Returns:
        requests.Response: response (.status_code, .text)
    """
    url = 'http://localhost:11434/api/chat' if url == 'localhost' else url
    json = {
        'model': model,
        'messages': messages,
        'keep_alive': -1 if keep_alive else '5m',
        'options': {
            'f16_kv': True,
            'low_vram': False,
            'use_mlock': False,
            'temperature': 0.7,
        },
        'stream': stream,
    }
    resp: requests.Response = requests.post(url=url, json=json)
    return resp


def widget_info_notification(body: str, icon: str = '✅', dur: float = 3.0):
    """Notification widget

    Args:
        body (str): message to display
        icon (str, optional): widget icon. Defaults to '✅'.
        dur (float, optional): display duration. Defaults to 3.
    """
    widget = st.success(body=body, icon=icon)
    time.sleep(dur)
    widget.empty()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ollama_base_url', help='ollama url:port', default='http://localhost:11434', type=str)
    args = parser.parse_args()

    # setup
    ollama_base_url = args.ollama_base_url
    ollama_identifier = 'qwen2.5:7b-instruct'
    ollama_stream = False

    # ---
    # SIDEBAR

    with st.sidebar:
        st.title('🤗💬 LLM Chat')
        st.markdown(f'''
        # About
        Chat with LLM model using Ollama interface.
        
        # Configuration
        ''')
        
        # flags
        ollama_stream = st.toggle('Stream response', help='Stream response continously or return it at once when done generating.')

        # select model option
        ollama_identifier = st.selectbox(
            'Select model you would like to chat with:',
            ('qwen2.5:0.5b-instruct', 'qwen2.5:7b-instruct', 'llama3.1:7b', 'phi3.5', 'gemma2:9b')
        )

        st.markdown(f'''
        ## Model description
        * Qwen2.5 and Gemma2 are particularly strong in emotional understanding making them suitable for conversational roles, like a psychologist or therapist roles.
        * Llama 3.1 offers a broad knowledge base, suitable for informative and insightful conversations.
        * Phi 3.5 excels in reasoning and analytical tasks but may be less focused on emotional nuances, but good for motivatiion and coaching.
        
        # Running:
        Using **{ollama_identifier}** with streaming **{"enabled" if ollama_stream else "disabled"}**.
        
        # Cache
        ''')
        if st.button('Delete messages', use_container_width=True, on_click=lambda: st.session_state.messages.clear()):
            widget_info_notification('Messages cleared!')

    # ---
    # MAIN WINDOW
    
    st.header("Chat here 💬")

    # init messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat
    if prompt := st.chat_input("What can I help you with?"):
        # user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # llm: generate response
        with st.chat_message("assistant"):
            response = ollama_chat(
                messages=st.session_state.messages,
                model=ollama_identifier,
                url=ollama_base_url,
                stream=ollama_stream,
                keep_alive=False,
            )
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    print(st.session_state.messages)







