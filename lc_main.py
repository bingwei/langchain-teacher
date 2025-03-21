import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, AIMessage
from langsmith import Client
from langchain.callbacks.base import BaseCallbackHandler
from vector_store import batch_import_guides, load_faiss
from langchain.chains import LLMChain
from get_prompt import load_prompt, load_prompt_with_questions
from user_progress import save_progress, load_progress
import os
from loguru import logger

st.set_page_config(page_title="LangChain: Getting Started Class", page_icon="🦜")
st.title("🦜 LangChain: Getting Started Class")
button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Lesson selection dictionary
lesson_guides = {
    "Lesson 1: Getting Started with LangChain": {
        "description": "This lesson covers the basics of getting started with LangChain."
    },
    "Lesson 2: Prompts": {
        "description": "This lesson focuses on prompts and their usage."
    },
    "Lesson 3: Language Models": {
        "description": "This lesson provides an overview of language models."
    },
    "Lesson 4: Memory": {
        "description": "This lesson is about Memory."
    },
    "Lesson 5: Chains": {
        "description": "This lesson provides information on Chains in LangChain, their types, and usage."
    },
    "Lesson 6: Retrieval": {
        "description": "This lesson provides information on indexing and retrieving information using LangChain."
    },
    "Lesson 7: Agents": {
        "description": "This lesson provides information on agents, tools, and toolkits."
    }
}

# Initialize LangSmith client
client = Client()

# Lesson selection sidebar
lesson_selection = st.sidebar.selectbox("Select Lesson", list(lesson_guides.keys()))

# Display lesson content and description based on selection
lesson_info = lesson_guides[lesson_selection]


# 加载指定目录下的所有文档
docs_dir = "/Users/aiden/wrk/python/langchain/docs/docs"
persist_path = "./faiss_index"
# 尝试加载本地存储
db = load_faiss(persist_path, docs_dir)


lesson_description = lesson_info["description"]
logger.warning(f'faiss加载完成，lesson_description: {lesson_description}')
lesson_content = db.similarity_search(lesson_description, k=1)[0].page_content
logger.warning(f'faiss similarity_search结果: {lesson_content}')

# Radio buttons for lesson type selection
lesson_type = st.sidebar.radio("Select Lesson Type", ["Instructions based lesson", "Interactive lesson with questions"])

# Clear chat session if dropdown option or radio button changes
if st.session_state.get("current_lesson") != lesson_selection or st.session_state.get(
        "current_lesson_type") != lesson_type:
    st.session_state["current_lesson"] = lesson_selection
    st.session_state["current_lesson_type"] = lesson_type
    st.session_state["messages"] = [AIMessage(
        content="Welcome! This short course will help you get started with LangChain. Let me know when you're all set to jump in!")]

# Display lesson name and description
st.markdown(f"**{lesson_selection}**")
st.write(lesson_description)


# Message handling and interaction
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        # model = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-3.5-turbo-16k")
        model = OllamaLLM(model="qwen2.5:7b", callbacks=[stream_handler], streaming=True,
                          model_kwargs={"temperature": 0.7})

        if lesson_type == "Instructions based lesson":
            prompt_template = load_prompt(content=lesson_content)
        else:
            prompt_template = load_prompt_with_questions(content=lesson_content)

        chain = LLMChain(prompt=prompt_template, llm=model)

        response = chain(
            {"input": prompt, "chat_history": st.session_state.messages[-20:]},
            include_run_info=True,
            tags=[lesson_selection, lesson_type]
        )
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=response[chain.output_key]))
        save_progress("default_user", lesson_selection, True)
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
