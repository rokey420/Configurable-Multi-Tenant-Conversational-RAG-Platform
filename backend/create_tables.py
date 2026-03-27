import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL missing in .env")

def column_exists(cur, table: str, column: str) -> bool:
    cur.execute("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_name=%s AND column_name=%s
    """, (table, column))
    return cur.fetchone() is not None

try:
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    system_prompt TEXT DEFAULT '',
                    pinecone_namespace TEXT UNIQUE NOT NULL,
                    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS topic_members (
                    id SERIAL PRIMARY KEY,
                    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('admin','employee')),
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(topic_id, user_id)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
                    prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # add topic_id to chat_history if missing
            if not column_exists(cur, "chat_history", "topic_id"):
                cur.execute("ALTER TABLE chat_history ADD COLUMN topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    filename TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # add topic_id to documents if missing
            if not column_exists(cur, "documents", "topic_id"):
                cur.execute("ALTER TABLE documents ADD COLUMN topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL;")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_user_topic ON chat_history(user_id, topic_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_members_topic_user ON topic_members(topic_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_topic ON documents(topic_id);")
        
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_sessions (
                    token UUID PRIMARY KEY,
                    admin_id INTEGER NOT NULL REFERENCES admins(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_sessions_admin ON admin_sessions(admin_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_sessions_expires ON admin_sessions(expires_at);")

        conn.commit()

    print("Tables created/updated successfully")

except Exception as e:
    print(" Error:", e)
    raise