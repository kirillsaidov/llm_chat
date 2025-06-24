# module chatui

# system
import os
import re
import sys
import time
import uuid
from datetime import datetime

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
        messages (str): list of messages
        ollama_identifier (str): ollama model name
        ollama_stream (bool, optional): stream response. Defaults to False.
        ollama_keep_alive (int | str, optional): keep model loaded in memory [ -1: keep in memory, 5m: keep for 5 minutes]. Defaults to -1.
        ollama_options (OllamaOptions, optional): ollama options. Defaults to OllamaOptions(temperature=0.7, low_vram=False, use_mlock=False, f16_kv=True).

    Returns:
        dict: response {'response': 'llm response'}
    """
    response = ollama_client.chat(model=ollama_identifier, messages=messages, options=ollama_options, keep_alive=ollama_keep_alive, stream=ollama_stream)
    return response


def llm_ollama_generate_title(
    ollama_client: OllamaClient, 
    messages: list[dict], 
    ollama_identifier: str, 
    ollama_stream: bool = False,
    ollama_keep_alive: int | str = -1,
    ollama_options: OllamaOptions = OllamaOptions(temperature=1.0, low_vram=False, use_mlock=False, f16_kv=True, num_predict=50)
) -> str:
    """Generate ollama response

    Args:
        ollama_client (OllamaClient): ollama client
        messages (str): list of messages
        ollama_identifier (str): ollama model name
        ollama_stream (bool, optional): stream response. Defaults to False.
        ollama_keep_alive (int | str, optional): keep model loaded in memory [ -1: keep in memory, 5m: keep for 5 minutes]. Defaults to -1.
        ollama_options (OllamaOptions, optional): ollama options. Defaults to OllamaOptions(temperature=0.7, low_vram=False, use_mlock=False, f16_kv=True).

    Returns:
        str: response
    """
    messages_text = '\n'.join(
        f'{message["role"]}: {message["content"]}'
        for message in messages
        if message["role"] != "system"
    )
    
    resp = ollama_client.generate(
        model=ollama_identifier,
        prompt=messages_text,
        system="""You are a chat title generator. Your task is to create a concise, descriptive title for the conversation.

Requirements:
- Maximum 12 words (not 30 words - titles should be brief)
- Focus on the main topic or question discussed
- Use clear, specific language
- Avoid generic words like "chat", "conversation", "discussion"
- No punctuation marks or special characters
- Return ONLY the title, nothing else
- Do not include explanations, recommendations, or additional text

Examples:
- "Python data visualization help" 
- "React component debugging issue"
- "Travel planning for Japan"
- "SQL query optimization tips"

Generate a title that captures the essence of the provided chat conversation.""".strip(),
        options=ollama_options,
        keep_alive=ollama_keep_alive,
        stream=ollama_stream,
    )
    
    return resp['response'].split('</think>')[-1].strip()


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


class ChatManager:
    """MongoDB chat management"""
    
    def __init__(self, mongo_uri: str, mongo_db: str, mongo_collection: str):
        self.client = None
        self.reconnect(mongo_uri, mongo_db, mongo_collection)
        
    
    def reconnect(self, mongo_uri: str, mongo_db: str, mongo_collection: str):
        if mongo_uri: self.close()
        self.client = pymongo.MongoClient(mongo_uri)
        self.collection = self.client[mongo_db][mongo_collection]

    
    def close(self):
        if self.client: self.client.close()
        
    
    def generate_chat_title(self, messages: list) -> str:
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'][:50]
                if len(msg['content']) > 50:
                    content += '...'
                return content
        return f'Chat {datetime.now().isoformat()}'
    
    
    def list_chats(self) -> list:
        return list(self.collection.find())
    
    
    def save_chat(self, chat_id: str, messages: list, system_prompt: str, title: str = None):
        chat_data = {
            'id': chat_id,
            'title': title if title else self.generate_chat_title(messages),
            'messages': messages,
            'system_prompt': system_prompt,
            'updated_at': datetime.now().isoformat(),
        }
        
        # check if it exists: update or insert
        if self.collection.find_one({'id': chat_id}):
            chat_data.pop('title', None)
            self.collection.update_one({'id': chat_id}, {'$set': chat_data})
        else:
            chat_data['created_at'] = datetime.now().isoformat()
            self.collection.insert_one(chat_data)
    
    
    def load_chat(self, chat_id: str) -> dict | None:
        try: return self.collection.find_one({'id': chat_id})
        except: return None
    
    
    def delete_chat(self, chat_id: str) -> bool:
        try: return self.collection.delete_one({'id': chat_id}).deleted_count > 0
        except: return False
    


if __name__ == '__main__':
    # setup
    mongo_uri = MONGO_DEFAULT_URI
    mongo_db = MONGO_DEFAULT_DB
    mongo_collection = MONGO_DEFAULT_COLLECTION
    ollama_base_url = OLLAMA_DEFAULT_URL
    ollama_identifier = 'qwen2.5:7b-instruct'
    ollama_stream = False
    ollama_client = OllamaClient(ollama_base_url)
    enable_temporary_chat = False
    
    # create chat manager
    chat_manager = ChatManager(mongo_uri, mongo_db, mongo_collection)
    
    # init session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = OLLAMA_DEFAULT_SYSTEM_PROMPT
    if 'temporary_chat_mode' not in st.session_state:
        st.session_state.temporary_chat_mode = enable_temporary_chat
        
    # reset chat
    def reset_chat():
        st.session_state.messages = []
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.system_prompt = OLLAMA_DEFAULT_SYSTEM_PROMPT
    
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
            enable_temporary_chat = st.toggle('Enable temporary chat', help='This chat won\'t be saved to history.', value=st.session_state.temporary_chat_mode)
            if enable_temporary_chat:                
                # clear chat
                if st.button('Clear messages', use_container_width=True, on_click=lambda: st.session_state.messages.clear(), icon='🗑️'):
                    widget_info_notification('Messages cleared!')
            else:
                # list chats
                chats = sorted(chat_manager.list_chats(), key=lambda x: datetime.fromisoformat(x['created_at']), reverse=True)
                chat_list = {f'{chat["title"]} ({chat["created_at"]})': chat['id'] for chat in chats}
                selected_chat = st.selectbox(
                    'Select a chat:',
                    options=['New Chat'] + list(chat_list.keys()),
                    index=0 if st.session_state.current_chat_id not in chat_list.values() else 
                          (list(chat_list.values()).index(st.session_state.current_chat_id) + 1 
                           if st.session_state.current_chat_id in chat_list.values() else 0)
                )
                if selected_chat == 'New Chat': reset_chat()
                
                # buttons
                col_load, col_delete = st.columns(2)
                with col_load:
                    if st.button('Load Chat', use_container_width=True, disabled=(selected_chat == 'New Chat')):
                        chat_id = chat_list[selected_chat]
                        loaded_chat = chat_manager.load_chat(chat_id)
                        if loaded_chat:
                            st.session_state.messages = loaded_chat['messages']
                            st.session_state.current_chat_id = chat_id
                            st.session_state.system_prompt = loaded_chat.get('system_prompt', OLLAMA_DEFAULT_SYSTEM_PROMPT)
                            st.rerun()
                        else:
                            st.error('Failed to load chat!')
                with col_delete:
                    if st.button('Delete Chat', use_container_width=True, disabled=(selected_chat == 'New Chat')):
                        chat_id = chat_list[selected_chat]
                        if chat_manager.delete_chat(chat_id):
                            if st.session_state.current_chat_id == chat_id: reset_chat()
                            st.success('Chat deleted!')
                            st.rerun()
                        else:
                            st.error('Failed to delete chat!')
            
            # clear chat if needed
            if enable_temporary_chat and enable_temporary_chat != st.session_state.temporary_chat_mode: reset_chat()
            st.session_state.temporary_chat_mode = enable_temporary_chat
        
        # llm settings
        with tab_settings:
            # configuration
            st.markdown('# Configuration')
            ollama_stream = st.toggle('Stream response', help='Stream response continuously or return it at once when done generating.', value=True)
            render_markdown = st.toggle('Render markdown', help='Display text markdown.', value=True)
            generate_chat_title_with_llm = st.toggle('Generate chat title', help='Auto-generate chat title with LLM from it\'s content.', value=True)
            
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
            ollama_system_prompt = st.text_area('Instruction', st.session_state.system_prompt, help='System prompt.')

            # advanced settings
            show_advanced_settings = st.checkbox('Show advanced settings')
            if show_advanced_settings:
                st.markdown('### Mongo')
                mongo_uri = st.text_input('URI', value=MONGO_DEFAULT_URI)
                mongo_db = st.text_input('Database', value=MONGO_DEFAULT_DB)
                mongo_collection = st.text_input('Collection', value=MONGO_DEFAULT_COLLECTION)
                
                # update chat manager db connection
                chat_manager.reconnect(mongo_uri, mongo_db, mongo_collection)
                
                st.markdown('### Ollama')
                ollama_base_url = st.text_input('URL', value=OLLAMA_DEFAULT_URL)

            st.markdown(f'''
            # Running:
            Using **{ollama_identifier}** with streaming **{"enabled" if ollama_stream else "disabled"}**.
            ''')

    # ---
    # MAIN WINDOW
    
    st.header('Chat here 💬')

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

        # save chat
        if not enable_temporary_chat:
            try:
                chat_manager.save_chat(
                    chat_id=st.session_state.current_chat_id,
                    messages=st.session_state.messages,
                    system_prompt=st.session_state.system_prompt,
                    title=llm_ollama_generate_title(
                        ollama_client=ollama_client,
                        messages=st.session_state.messages,
                        ollama_identifier=ollama_identifier,
                        ollama_options=ollama_options,
                    ) if generate_chat_title_with_llm and selected_chat == 'New Chat' else None,
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save chat: {e}")

