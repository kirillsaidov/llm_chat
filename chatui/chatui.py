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
OLLAMA_DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'


def ollama_chat(
    ollama_client: OllamaClient, 
    messages: list[dict], 
    ollama_identifier: str, 
    ollama_stream: bool = False,
    ollama_keep_alive: int | str = -1,
    ollama_options: OllamaOptions = OllamaOptions(temperature=1.0, low_vram=False, use_mlock=False, f16_kv=True, num_ctx=16384)
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


def extract_thinking_and_content(text: str):
    """Extract thinking part and main content from LLM response"""
    # look for <think>...</think> pattern
    thinking_pattern = r'<think>(.*?)</think>'
    match = re.search(thinking_pattern, text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        # remove the thinking part from the original text
        main_content = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
        return thinking_content, main_content
    else:
        # thinking in progress
        if '<think>' in text:
            return text.split('<think>')[-1], None
        
        # no thinking part found, return empty thinking and full content
        return None, text


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
        render_markdown = st.toggle('Render markdown', help='Display text markdown.', value=True)

        # select model option
        ollama_identifier = st.selectbox(
            'Select model you would like to chat with:',
            ('llama3.1:8b', 'gemma2:9b', 'gemma2:27b', 'qwen2.5:7b-instruct', 'qwen3:14b', 'falcon:7b', 'falcon2:11b', 'other')
        )
        if ollama_identifier == 'other': ollama_identifier = st.text_input('Enter model name manually:')
        
        # init ollama options
        ollama_options = OllamaOptions(
            temperature=st.number_input('Creativity (0 - logical, 1 - creative)', value=1.0, min_value=0.0, max_value=1.0, step=0.1),
            low_vram=False,
            use_mlock=False,
            f16_kv=True,
        )
        
        # set prompt
        ollama_system_prompt = st.text_area('Instruction', OLLAMA_DEFAULT_SYSTEM_PROMPT)

        st.markdown(f'''
        # Running:
        Using **{ollama_identifier}** with streaming **{"enabled" if ollama_stream else "disabled"}**.
        
        # Cache
        ''')
        if st.button('Delete messages', use_container_width=True, on_click=lambda: st.session_state.messages.clear()):
            widget_info_notification('Messages cleared!')

    # ---
    # MAIN WINDOW
    
    st.header('Chat here 💬')

    # init messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # display history
    st_content_display = st.markdown if render_markdown else st.text
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            thinking, content = extract_thinking_and_content(message['content'])
            if thinking:
                with st.expander('💭 Show thinking process', expanded=False):
                    st.markdown(thinking)
            st_content_display(content)

    # chat
    if prompt := st.chat_input('What can I help you with?'):
        # ensure system prompt
        if not len(st.session_state.messages):
            st.session_state.system_prompt = ollama_system_prompt
            st.session_state.messages.append({
                'role': 'system', 
                'content': ollama_system_prompt
            })
        elif st.session_state.system_prompt != ollama_system_prompt:
            st.session_state.system_prompt = ollama_system_prompt
            st.session_state.messages.append({
                'role': 'system', 
                'content': ollama_system_prompt
            })
        
        # user
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # llm: generate response
        with st.chat_message('assistant'):
            response = ollama_chat(
                ollama_client=ollama_client,
                messages=st.session_state.messages,
                ollama_identifier=ollama_identifier,
                ollama_stream=ollama_stream,
                ollama_options=ollama_options,
            )
            
            # streaming
            response_text = ''
            if ollama_stream:
                response_placeholder = st.empty()
                for chunk in response:
                    response_text += chunk['message']['content']
                    thinking, content = extract_thinking_and_content(response_text)
                    with response_placeholder.container():
                        if thinking:
                            with st.expander('💭 Show thinking process' if thinking and content else '💭 Thinking...', expanded=True):
                                st.markdown(thinking)
                        if content:
                            st_content_display(content)
            else:
                response_text = response['message']['content']
                thinking, content = extract_thinking_and_content(response_text)
                if thinking:
                    with st.expander('💭 Show thinking process', expanded=False):
                        st.markdown(thinking)
                st_content_display(content)
        st.session_state.messages.append({'role': 'assistant', 'content': response_text})



