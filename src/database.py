import sqlite3

from src.paths import SQLITE_DB_PATH, ensure_app_directories


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_text TEXT,
    type TEXT DEFAULT 'document',
    chat_id INTEGER,
    FOREIGN KEY (chat_id) REFERENCES chat(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chat_id) REFERENCES chat(id)
);
"""


def connect_db():
    ensure_app_directories()
    return sqlite3.connect(SQLITE_DB_PATH)


def initialize_database():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        conn.commit()


def create_chat(title):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat (title) VALUES (?)", (title,))
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return chat_id


def list_chats():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat ORDER BY created_at DESC")
    chats = cursor.fetchall()
    conn.close()
    return chats


def read_chat(chat_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat WHERE id = ?", (chat_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def update_chat(chat_id, new_title):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (new_title, chat_id),
    )
    conn.commit()
    conn.close()


def delete_chat(chat_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def create_source(name, source_text, chat_id, source_type="document"):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sources (name, source_text, chat_id, type) VALUES (?, ?, ?, ?)",
        (name, source_text, chat_id, source_type),
    )
    conn.commit()
    conn.close()


def read_source(source_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sources WHERE id = ?", (source_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def update_source(source_id, new_name, new_source_text):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sources SET name = ?, source_text = ? WHERE id = ?",
        (new_name, new_source_text, source_id),
    )
    conn.commit()
    conn.close()


def list_sources(chat_id, source_type=None):
    conn = connect_db()
    cursor = conn.cursor()
    if source_type:
        cursor.execute(
            "SELECT * FROM sources WHERE chat_id = ? AND type = ?",
            (chat_id, source_type),
        )
    else:
        cursor.execute("SELECT * FROM sources WHERE chat_id = ?", (chat_id,))
    sources = cursor.fetchall()
    conn.close()
    return sources


def delete_source(source_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    conn.commit()
    conn.close()


def create_message(chat_id, sender, content):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
        (chat_id, sender, content),
    )
    conn.commit()
    conn.close()


def get_messages(chat_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sender, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
        (chat_id,),
    )
    messages = cursor.fetchall()
    conn.close()
    return messages


def delete_messages(chat_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()
