import os
from pathlib import Path
from typing import Optional, Dict, Any

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL in .env")

GENERAL_NAMESPACE = "general"
GENERAL_TOPIC_NAME = "General"
GENERAL_TOPIC_PROMPT = "You are a friendly general assistant."


def get_conn():
    return psycopg2.connect(DB_URL)


def ensure_general_topic(conn) -> int:
    """Create 'General' topic once if missing; return its topic_id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM topics WHERE pinecone_namespace=%s", (GENERAL_NAMESPACE,))
        row = cur.fetchone()
        if row:
            return int(row[0])

        cur.execute(
            """
            INSERT INTO topics(name, system_prompt, pinecone_namespace, created_by)
            VALUES(%s,%s,%s,%s)
            RETURNING id
            """,
            (GENERAL_TOPIC_NAME, GENERAL_TOPIC_PROMPT, GENERAL_NAMESPACE, None),
        )
        return int(cur.fetchone()[0])


def get_or_create_user(username: str) -> int:
    with get_conn() as conn:
        ensure_general_topic(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s", (username,))
            row = cur.fetchone()
            if row:
                return int(row[0])
            cur.execute("INSERT INTO users(username) VALUES(%s) RETURNING id", (username,))
            user_id = int(cur.fetchone()[0])
        conn.commit()
        return user_id


def save_chat(
    user_id: int,
    topic_id: int,
    prompt: str,
    answer: str,
    session_id: Optional[str] = None,
):
    """Always save with topic_id (General or selected topic)."""
    with get_conn() as conn:
        ensure_general_topic(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history(user_id, topic_id, session_id, prompt, answer)
                VALUES(%s,%s,%s,%s,%s)
                """,
                (user_id, topic_id, session_id, prompt, answer),
            )
        conn.commit()


def require_topic_access(conn, user_id: int, topic_id: int) -> Dict[str, Any]:
    """For non-General topics: ensure user is member; return topic info."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM topics WHERE pinecone_namespace=%s", (GENERAL_NAMESPACE,))
        general = cur.fetchone()
        general_id = int(general[0]) if general else None

        if general_id is not None and topic_id == general_id:
            return {
                "topic_id": topic_id,
                "name": GENERAL_TOPIC_NAME,
                "system_prompt": GENERAL_TOPIC_PROMPT,
                "namespace": GENERAL_NAMESPACE,
                "role": "employee",
            }

        cur.execute(
            """
            SELECT t.id, t.name, t.system_prompt, t.pinecone_namespace, tm.role
            FROM topics t
            JOIN topic_members tm ON tm.topic_id = t.id
            WHERE t.id=%s AND tm.user_id=%s
            """,
            (topic_id, user_id),
        )
        row = cur.fetchone()
        if not row:
            raise PermissionError("You do not have access to this topic.")

        return {
            "topic_id": row[0],
            "name": row[1],
            "system_prompt": row[2] or "",
            "namespace": row[3],
            "role": row[4],
        }