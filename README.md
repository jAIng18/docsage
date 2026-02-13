# Streamlit-Naive-RAG-DocSage

DocSage is a lightweight document-aware chat application built with Streamlit, SQLite, and LangChain.
It allows you to create multiple chats, upload documents or links as knowledge sources, and ask questions that are answered strictly using the provided context.

The system uses vector embeddings and retrieval to ground answers in your documents.

>> NOTE: This repo is inspired by page Ngonidzashe Nzenze's post. To understand in details, checkout [his page](https://dev.to/ngonidzashe/doc-sage-create-a-smart-rag-app-with-langchain-and-streamlit-4lin)

## Project Overview
At a high level, the project consists of four main parts:
1. Streamlit UI [chats.py](chats.py)  
Handles the web interface, chat flow, document uploads, and link ingestion.

2. Database Schema [create_relational_db.py](create_relational_db.py)  
Initializes the SQLite database with tables for chats, messages, and sources.

3. Database Access Layer [db.py](db.py)  
Provides clean CRUD functions to manage chats, messages, and sources.

4. Vector & RAG Logic [vector_functions.py](db.py)  
Handles document loading, embedding, vector storage, retrieval, and answer generation using OpenAI models.

## Application Flow

```mermaid
flowchart LR
    subgraph Input["Input"]
        User([User])
        Doc[Documents]
        Link[URLs]
    end

    subgraph App["DocSage"]
        UI[Streamlit UI]
        DB[(SQLite)]
        Vector[(ChromaDB)]
    end

    subgraph AI["OpenAI"]
        Embed[Embeddings]
        GPT[GPT-4o-mini]
    end

    User --> UI
    Doc --> UI
    Link --> UI
    
    UI <--> DB
    UI --> Vector
    Vector <--> Embed
    
    UI --> GPT
    Vector --> GPT
    GPT --> UI
```

1. A user creates a new chat.
2. Documents or web links are uploaded and stored as context.
3. Text content is embedded and stored in a Chroma vector database.
4. When a question is asked:
    - Relevant chunks are retrieved using similarity search.
    - The LLM generates an answer using only the retrieved context
7. All chats and messages are stored persistently in SQLite.

## Setup Guide
1. Clone the repo
2. Create venv (optional)
3. Install Dependencies
```shell
pip install -r requirements.txt
```
4. Create an OpenAI API Key (must do)
5. Initialize the Database
```shell
python create_relational_db.py
```
6. Run the application
```shell
streamlit run chats.py
```
## Notes and Limitations
- Answers are generated only from uploaded documents and links.
- If no context exists, the assistant will not answer.
- Each chat has its own isolated vector store.
- The vector database is stored locally under the persist/ directory.

## Conclusion
DocSage is a clean, minimal example of a document-grounded chat system using modern LLM tooling.  
It is well-suited for learning RAG concepts, prototyping internal knowledge assistants, or extending into a production-ready system.

## Reference
Tips to improve RAG: [Link page](https://www.chitika.com/open-source-models-rag/)
Create a Smart RAG App with LangChain and Streamlit: [Link page](https://dev.to/ngonidzashe/doc-sage-create-a-smart-rag-app-with-langchain-and-streamlit-4lin)