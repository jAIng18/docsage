import os

import environ
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from src.paths import PROJECT_ROOT, VECTOR_STORE_DIR, ensure_app_directories


env = environ.Env()
environ.Env.read_env(str(PROJECT_ROOT / ".env"))

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=env("OPENAI_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    api_key=env("OPENAI_API_KEY"),
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def collection_exists(collection_name: str) -> bool:
    ensure_app_directories()
    client = PersistentClient(path=str(VECTOR_STORE_DIR))
    collections = client.list_collections()
    return any(getattr(collection, "name", collection) == collection_name for collection in collections)


def load_document(file_path: str) -> list[Document]:
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".txt":
        loader = TextLoader(file_path)
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".csv":
        loader = CSVLoader(file_path)
    elif file_extension == ".html":
        loader = UnstructuredHTMLLoader(file_path)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return loader.load()


def create_collection(collection_name, documents):
    texts = text_splitter.split_documents(documents)
    ensure_app_directories()
    persist_directory = str(VECTOR_STORE_DIR)

    try:
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except Exception as error:
        print(f"Error creating collection: {error}")
        return None

    return vectordb


def load_collection(collection_name):
    ensure_app_directories()
    persist_directory = str(VECTOR_STORE_DIR)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectordb


def add_documents_to_collection(vectordb, documents):
    texts = text_splitter.split_documents(documents)
    vectordb.add_documents(texts)
    return vectordb


def load_retriever(collection_name, score_threshold: float = 0.6):
    vectordb = load_collection(collection_name)
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold},
    )
    return retriever


def generate_answer_from_context(retriever, question: str):
    message = """
    Answer this question using the provided context only.
    {question}

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    return rag_chain.invoke(question).content
