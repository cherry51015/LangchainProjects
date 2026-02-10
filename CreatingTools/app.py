import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
#from langchain.agents import initialize_agent,AgentType
#from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your groq API key:",type="password")


# 1. Create memory
#if "messages" not in st.session_state:
    #st.session_state["messages"]=[{"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}]

# 1. Create memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Show old messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# 3. Take new input
prompt=st.chat_input("Say something")

if prompt:
      # 4. Save user message
    st.session_state.messages.append({"role":"user","content":prompt})

      # 5. Display user message
    st.chat_message("user").write(prompt)

      # 6. Create assistant reply
    reply="I heard u say:"+prompt

      # 7. Save assistant message
    st.session_state.messages.append({"role":"assistant","content":reply})

      # 8. Display assistant message
    st.chat_message("assistant").write(reply)