from __future__ import annotations
from datetime import datetime, timedelta
from uuid import uuid4

from jose import JWTError, jwt

from config.settings import settings


def create_access_token(user_id: str, tenant_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "exp": expire,
        "type": "access",
        "jti": str(uuid4()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def create_refresh_token(user_id: str) -> tuple[str, datetime]:
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid4()),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return token, expire


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except JWTError:
        raise ValueError("Invalid token")
