# Chat with any LLM using Ollama and WebUI interface.
WebUI for chatting with LLMs.

## Install
```sh
# clone repository
$ git clone https://github.com/kirillsaidov/llm_chat.git
$ cd llm_chat/

# install python dependencies
$ python3 -m venv venv && source ./venv/bin/activate
$ pip install -r requirements.txt
```

## Run
```sh
$ streamlit run chatui/chatui.py --server.port=8501
```
Output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://xxx.xxx.xxx.xxx:8501
```

# Available models:
You can change the `ollama_identifier = st.selectbox(...)` to add more models. Here is a list of available models:
* qwen2.5:3b-instruct
* qwen2.5:7b-instruct
* llama3.1:8b
* phi3.5
* gemma2:9b

## LICENSE
Unlicense. You can do whatever you want with the repo files.


