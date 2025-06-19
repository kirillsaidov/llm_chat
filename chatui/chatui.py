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

# mongo
import pymongo 

# constants
MONGO_DEFAULT_URI = 'mongodb://localhost:27017'
MONGO_DEFAULT_DB = 'llm_chat'
MONGO_DEFAULT_COLLECTION = 'chats'
OLLAMA_DEFAULT_URL = 'http://localhost:11434'
OLLAMA_DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'


def ollama_chat(
    ollama_client: OllamaClient, 
    messages: list[dict], 
    ollama_identifier: str, 
    ollama_stream: bool = False,
    ollama_keep_alive: int | str = -1,
    ollama_options: OllamaOptions = OllamaOptions(temperature=1.0, low_vram=False, use_mlock=False, f16_kv=True, num_ctx=32768)
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


def extract_thinking_and_content(text: str) -> tuple[str, str]:
    """Extract thinking part and main content from LLM response

    Args:
        text (str): LLM full response

    Returns:
        tuple[str, str]: thinking, content
    """
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
    

def mongo_reconnect(mongo_uri: str, mongo_client: pymongo.MongoClient = None) -> pymongo.MongoClient:
    """Reconnect to new mongo URI.

    Args:
        mongo_uri (str): new mongo uri
        mongo_client (pymongo.MongoClient): current mongo client to close. Defaults to None.

    Returns:
        pymongo.MongoClient: new mongo client
    """
    if mongo_client: mongo_client.close()
    return pymongo.MongoClient(mongo_uri)


class ChatManager:
    """MongoDB chat management"""
    pass 


if __name__ == '__main__':
    # setup
    mongo_uri = MONGO_DEFAULT_URI
    mongo_db = MONGO_DEFAULT_DB
    mongo_collection = MONGO_DEFAULT_COLLECTION
    mongo_client = mongo_reconnect(mongo_uri)
    ollama_base_url = OLLAMA_DEFAULT_URL
    ollama_identifier = 'qwen2.5:7b-instruct'
    ollama_stream = False
    ollama_client = OllamaClient(ollama_base_url)
    enable_temporary_chat = False
    
    # ---
    # SIDEBAR

    with st.sidebar:
        st.title('🤗💬 LLM Chat')
        
        # split into tabs
        tab_chat, tab_settings = st.tabs(['💾 Chat', '⚙️ Settings'])
        
        # chat management
        with tab_chat:
            st.markdown(f'''
            # About
            Chat with LLM model using Ollama interface.
            ''')
            
            # configuration
            enable_temporary_chat = st.toggle('Enable temporary chat', help='This chat won\'t be saved to history.', value=False)
            if enable_temporary_chat:
                # close mongo connection
                if mongo_client: mongo_client.close()
                
                # clear chat
                if st.button('Clear messages', use_container_width=True, on_click=lambda: st.session_state.messages.clear(), icon='🗑️'):
                    widget_info_notification('Messages cleared!')
            else:
                pass
        
        # llm settings
        with tab_settings:
            # configuration
            st.markdown('# Configuration')
            ollama_stream = st.toggle('Stream response', help='Stream response continuously or return it at once when done generating.', value=True)
            render_markdown = st.toggle('Render markdown', help='Display text markdown.', value=True)

            # select model option
            st.markdown('### General')
            ollama_identifier = st.selectbox(
                'Select model you would like to chat with:',
                ('gemma2:9b', 'gemma2:27b', 'qwen2.5:7b-instruct', 'qwen3:14b', 'other'),
                help='Any model supported by Ollama.'
            )
            if ollama_identifier == 'other': ollama_identifier = st.text_input('Enter model name manually:')
            
            # init ollama options
            ollama_options = OllamaOptions(
                temperature=st.number_input('Temperature', value=1.0, min_value=0.0, max_value=1.0, step=0.1, help='0 - logical, 1 - creative'),
                low_vram=False,
                use_mlock=False,
                f16_kv=True,
            )
            ollama_system_prompt = st.text_area('Instruction', OLLAMA_DEFAULT_SYSTEM_PROMPT, help='System prompt.')

            # advanced settings
            show_advanced_settings = st.checkbox('Show advanced settings')
            if show_advanced_settings:
                st.markdown('### Mongo')
                mongo_uri = st.text_input('URI', value=MONGO_DEFAULT_URI)
                mongo_db = st.text_input('Database', value=MONGO_DEFAULT_DB)
                mongo_collection = st.text_input('Collection', value=MONGO_DEFAULT_COLLECTION)
                mongo_client = mongo_reconnect(mongo_uri, mongo_client)
                
                st.markdown('### Ollama')
                ollama_base_url = st.text_input('URL', value=OLLAMA_DEFAULT_URL)

            st.markdown(f'''
            # Running:
            Using **{ollama_identifier}** with streaming **{"enabled" if ollama_stream else "disabled"}**.
            ''')

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



