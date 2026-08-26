from datetime import datetime, timedelta, timezone
from typing import Optional, TypedDict

import bcrypt
import jwt

from .config import JWT_SECRET, SESSION_MAX_AGE_SECONDS


class SessionPayload(TypedDict):
    userId: int
    email: str


def create_session_token(user_id: int, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "userId": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(seconds=SESSION_MAX_AGE_SECONDS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_session_token(token: str) -> Optional[SessionPayload]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {"userId": payload["userId"], "email": payload["email"]}
    except jwt.PyJWTError:
        return None
    except KeyError:
        return None


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False
