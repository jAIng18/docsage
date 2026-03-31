import math
import time

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.database import (
    create_chat,
    create_message,
    create_source,
    delete_chat,
    delete_source,
    get_messages,
    list_chats,
    list_sources,
    read_chat,
)
from src.paths import TEMP_UPLOADS_DIR, ensure_app_directories
from src.rag_service import (
    add_documents_to_collection,
    collection_exists,
    create_collection,
    generate_answer_from_context,
    load_collection,
    load_document,
    load_retriever,
)


def chats_home():
    st.markdown("<h1 style='text-align: center;'>Doc Retrieval</h1>", unsafe_allow_html=True)

    with st.container(border=True):
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            chat_title = st.text_input(
                "Chat Title",
                placeholder="Enter Chat Title",
                key="chat_title",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Chat", type="primary"):
                if chat_title:
                    chat_id = create_chat(chat_title)
                    st.success(f"Created new chat: {chat_title}")
                    st.query_params.from_dict({"chat_id": chat_id})
                    st.rerun()
                else:
                    st.warning("Please enter a chat title")

    with st.container(border=True):
        st.subheader("Previous Chats")
        previous_chats = list_chats()

        chats_per_page = 5
        total_pages = max(1, math.ceil(len(previous_chats) / chats_per_page))

        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        start_idx = (st.session_state.current_page - 1) * chats_per_page
        end_idx = start_idx + chats_per_page

        for chat in previous_chats[start_idx:end_idx]:
            chat_id, chat_title = chat[0], chat[1]
            with st.container(border=True):
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])

                with col1:
                    st.markdown(f"**{chat_title}**")
                with col2:
                    if st.button("Open", key=f"open_{chat_id}"):
                        st.query_params.from_dict({"chat_id": chat_id})
                        st.rerun()

                with col3:
                    if st.button("Delete", key=f"delete_{chat_id}"):
                        delete_chat(chat_id)
                        st.success(f"Deleted chat: {chat_title}")
                        st.rerun()

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()


def stream_response(response):
    """Stream a response word by word."""
    for word in response.split():
        yield f"{word} "
        time.sleep(0.05)


def render_sidebar_item(label: str, button_key: str, help_text: str, is_link: bool = False) -> bool:
    with st.container(border=True):
        content_col, action_col = st.columns([0.84, 0.16])

        with content_col:
            if is_link:
                st.markdown(f"[{label}]({label})")
            else:
                st.write(label)

        with action_col:
            return st.button("X", key=button_key, help=help_text, use_container_width=True)


def save_documents_for_chat(chat_id: int, documents: list[Document]):
    collection_name = f"chat_{chat_id}"
    if collection_exists(collection_name):
        vectordb = load_collection(collection_name)
        add_documents_to_collection(vectordb, documents)
    else:
        create_collection(collection_name, documents)


def chat_page(chat_id):
    chat = read_chat(chat_id)
    if not chat:
        st.error("Chat not found")
        return

    messages = get_messages(chat_id)

    if messages:
        for sender, content in messages:
            if sender == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            elif sender == "ai":
                with st.chat_message("assistant"):
                    st.markdown(content)
    else:
        st.write("No messages yet. Start the conversation!")

    prompt = st.chat_input("Type your message here...")
    if prompt:
        create_message(chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        collection_name = f"chat_{chat_id}"
        has_context = bool(list_sources(chat_id)) and collection_exists(collection_name)
        retriever = load_retriever(collection_name=collection_name) if has_context else None

        response = (
            generate_answer_from_context(retriever, prompt)
            if retriever
            else "I need some context to answer that question."
        )

        create_message(chat_id, "ai", response)

        with st.chat_message("assistant"):
            st.write_stream(stream_response(response))

        st.rerun()

    with st.sidebar:
        if st.button("Back to Chats"):
            st.query_params.clear()
            st.rerun()

        st.subheader(f"{chat[1]}")
        st.subheader("Documents")

        documents = list_sources(chat_id, source_type="document")
        if documents:
            for doc in documents:
                doc_id = doc[0]
                doc_name = doc[1]
                if render_sidebar_item(
                    label=doc_name,
                    button_key=f"delete_doc_{doc_id}",
                    help_text="Remove document",
                ):
                    delete_source(doc_id)
                    st.success(f"Deleted document: {doc_name}")
                    st.rerun()
        else:
            st.write("No documents uploaded.")

        uploaded_file = st.file_uploader("Upload Document", key="file_uploader")
        if uploaded_file:
            with st.spinner("Processing document..."):
                ensure_app_directories()
                temp_file_path = TEMP_UPLOADS_DIR / uploaded_file.name
                with temp_file_path.open("wb") as file_handle:
                    file_handle.write(uploaded_file.getbuffer())

                documents = load_document(str(temp_file_path))
                save_documents_for_chat(chat_id, documents)
                create_source(uploaded_file.name, "", chat_id, source_type="document")

                temp_file_path.unlink(missing_ok=True)
                st.session_state.pop("file_uploader", None)
                st.rerun()

        st.subheader("Links")

        links = list_sources(chat_id, source_type="link")
        if links:
            for link in links:
                link_id = link[0]
                link_url = link[1]
                if render_sidebar_item(
                    label=link_url,
                    button_key=f"delete_link_{link_id}",
                    help_text="Remove link",
                    is_link=True,
                ):
                    delete_source(link_id)
                    st.success(f"Deleted link: {link_url}")
                    st.rerun()
        else:
            st.write("No links added.")

        new_link = st.text_input("Add a link", key="new_link")
        if st.button("Add Link", key="add_link_btn"):
            if new_link:
                with st.spinner("Processing link..."):
                    try:
                        headers = {
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/86.0.4240.111 Safari/537.36"
                            )
                        }
                        response = requests.get(new_link, headers=headers, timeout=20)
                        soup = BeautifulSoup(response.text, "html.parser")

                        if response.status_code == 200 and soup.text.strip():
                            link_content = soup.get_text(separator="\n")
                        else:
                            st.toast(
                                "Unable to retrieve content from the link. It may be empty or inaccessible.",
                            )
                            return

                        documents = [
                            Document(
                                page_content=link_content,
                                metadata={"source": new_link},
                            )
                        ]
                        save_documents_for_chat(chat_id, documents)

                        create_source(new_link, "", chat_id, source_type="link")
                        st.success(f"Added link: {new_link}")
                        st.rerun()
                    except Exception as error:
                        st.toast(
                            f"Failed to fetch content from the link: {error}",
                        )
            else:
                st.toast("Please enter a link")


def main():
    """Main entry point for the chat application."""
    ensure_app_directories()
    query_params = st.query_params
    if "chat_id" in query_params:
        chat_id = int(query_params["chat_id"])
        chat_page(chat_id)
    else:
        chats_home()


if __name__ == "__main__":
    main()
