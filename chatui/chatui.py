# module chatui

# system
import os
import re
import sys
import time
import argparse
from enum import Enum

# webui
import streamlit as st

# ollama
from ollama import (
    Client as OllamaClient,
    Options as OllamaOptions,
)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ollama_base_url', help='ollama url:port', default='http://localhost:11434', type=str)
    parser.add_argument('--ollama_id', help='ollama model identifier (name)', default='qwen2.5:0.5b-instruct', type=str)
    parser.add_argument('--ollama_stream', help='stream ollama response', action='store_true')
    parser.add_argument('--verbose', help='verbose output', action='store_true')
    args = parser.parse_args()

    # setup
    ollama_base_url = args.ollama_base_url
    ollama_id = args.ollama_id
    ollama_stream = args.ollama_stream
    verbose = args.verbose

    # ---
    # SIDEBAR

    with st.sidebar:
        st.title('🤗💬 LLM Chat')
        st.markdown(f'''
        # About
        Chat with LLM model using Ollama interface.

        # Running:
        Using {ollama_id} with streaming={ollama_stream}.
        ''')

    # ---
    # MAIN WINDOW

    # init messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat
    if prompt := st.chat_input("What is up?"):
        # user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # llm: generate response
        with st.chat_message("assistant"):
            response = ollama_chat(
                messages=st.session_state.messages,
                model=ollama_id,
                url=ollama_base_url,
                stream=ollama_stream,
            )
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    print(st.session_state.messages)







