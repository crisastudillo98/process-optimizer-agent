from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from storage.database import get_db
from storage.models import User
from auth.jwt import decode_token

bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Wrong token type")

    user = db.query(User).filter(
        User.id == payload["sub"],
        User.is_active == True,
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Admin required")
    return current_user
