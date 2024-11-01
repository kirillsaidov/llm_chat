# module chatui

# system
import os
import re
import sys
import time
import argparse
import requests

# ollama
from ollama import Client as OllamaClient, Options as OllamaOptions

# webui
import streamlit as st

# constants
OLLAMA_DEFAULT_SYSTEM_PROMPT = '\
You are helpful and empathetic assistant skilled in psychological counseling. \
Your primary goal is to help the patient to identify his problems or issues, work out his or her problems to improve his or her mental wellbeing.'


def ollama_chat(
    ollama_client: OllamaClient, 
    messages: list[dict], 
    ollama_identifier: str, 
    ollama_stream: bool = False,
    ollama_keep_alive: int | str = -1,
    ollama_options: OllamaOptions = OllamaOptions(temperature=1.0, low_vram=False, use_mlock=False, f16_kv=True)
) -> dict:
    """Generate LLM response

    Args:
        ollama_client (OllamaClient): ollama client
        user_prompt (str): user prompt
        system_prompt (str): system prompt
        ollama_identifier (str): ollama model name
        ollama_stream (bool, optional): stream response. Defaults to False.
        ollama_keep_alive (int | str, optional): keep model loaded in memory [ -1: keep in memory, 5m: keep for 5 minutes]. Defaults to -1.
        ollama_options (OllamaOptions, optional): ollama options. Defaults to OllamaOptions(temperature=0.7, low_vram=False, use_mlock=False, f16_kv=True).

    Returns:
        dict: response metadata
    """
    response = ollama_client.chat(model=ollama_identifier, messages=messages, options=ollama_options, keep_alive=ollama_keep_alive, stream=ollama_stream)
    return response


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
    
    # init ollama
    ollama_client = OllamaClient()

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
        ollama_stream = st.toggle('Stream response', help='Stream response continuously or return it at once when done generating.', value=True)

        # select model option
        ollama_identifier = st.selectbox(
            'Select model you would like to chat with:',
            ('qwen2.5:3b-instruct', 'qwen2.5:7b-instruct', 'llama3.1:8b', 'phi3.5', 'gemma2:9b')
        )
        
        # set prompt
        ollama_system_prompt = st.text_area("Instruction", OLLAMA_DEFAULT_SYSTEM_PROMPT)

        st.markdown(f'''
        # Running:
        Using **{ollama_identifier}** with streaming **{"enabled" if ollama_stream else "disabled"}**.

        ## Model description
        * Qwen2.5 and Gemma2 are particularly strong in emotional understanding making them suitable for conversational roles, like a psychologist or therapist roles.
        * Llama 3.1 offers a broad knowledge base, suitable for informative and insightful conversations.
        * Phi 3.5 excels in reasoning and analytical tasks but may be less focused on emotional nuances, but good for motivatiion and coaching.
        
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
        # ensure system prompt
        if not len(st.session_state.messages):
            st.session_state.system_prompt = ollama_system_prompt
            st.session_state.messages.append({
                "role": "system", 
                "content": ollama_system_prompt
            })
        elif st.session_state.system_prompt != ollama_system_prompt:
            st.session_state.system_prompt = ollama_system_prompt
            st.session_state.messages.append({
                "role": "system", 
                "content": ollama_system_prompt
            })
        
        # user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # llm: generate response
        with st.chat_message("assistant"):
            response = ollama_chat(
                ollama_client=ollama_client,
                messages=st.session_state.messages,
                ollama_identifier=ollama_identifier,
                ollama_stream=ollama_stream,
            )
            
            # streaming
            response_text = ''
            if ollama_stream:
                response_placeholder = st.empty()
                with response_placeholder.container():
                    for chunk in response:
                        response_text += chunk['message']['content']
                        response_placeholder.markdown(response_text)
            else:
                response_text = response['message']['content']
                st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})



