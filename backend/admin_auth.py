import uuid
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

SESSION_TTL_HOURS = 24

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))

def new_token() -> uuid.UUID:
    return uuid.uuid4()

def expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)

def require_admin(cur, token: str) -> int:
    """
    Validates admin session token. Returns admin_id if valid, else raises.
    """
    try:
        tok = uuid.UUID(token)
    except Exception:
        raise ValueError("Invalid admin token")

    cur.execute("""
        SELECT admin_id
        FROM admin_sessions
        WHERE token=%s AND expires_at > (NOW() AT TIME ZONE 'UTC')
    """, (str(tok),))
    row = cur.fetchone()
    if not row:
        raise ValueError("Admin session expired or invalid")
    return int(row[0])