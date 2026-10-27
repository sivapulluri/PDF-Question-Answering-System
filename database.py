import sqlite3
from datetime import datetime

DB_NAME = "chat_history.db"


def init_db():
    """
    Initialize SQLite database and chat_logs table if not already created.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    pdf_name TEXT,
                    sources TEXT,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
    except Exception as e:
        print(f"Database initialization warning: {e}")


def save_chat(username, question, answer, pdf_name, sources, status):
    """
    Persist chat interaction record to SQLite database.
    """
    try:
        init_db()
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("""
                INSERT INTO chat_logs (
                    username, question, answer, pdf_name,
                    sources, status, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                username,
                question,
                answer,
                pdf_name,
                sources,
                status,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
    except Exception as e:
        print(f"Error saving chat log to database: {e}")
        raise


def get_all_chats():
    """
    Retrieve all chat history records ordered by newest first.
    Returns empty list if table is empty or on error.
    """
    try:
        init_db()
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.execute("""
                SELECT username, question, answer, pdf_name,
                       sources, status, timestamp
                FROM chat_logs
                ORDER BY id DESC
            """)
            return cursor.fetchall()
    except Exception as e:
        print(f"Error reading chat logs from database: {e}")
        return []