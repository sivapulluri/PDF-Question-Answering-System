import sqlite3
from datetime import datetime

DB_NAME = "chat_history.db"


def init_db():
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


def save_chat(username, question, answer, pdf_name, sources, status):
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


def get_all_chats():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT username, question, answer, pdf_name,
                   sources, status, timestamp
            FROM chat_logs
            ORDER BY id DESC
        """).fetchall()