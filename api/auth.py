from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from auth.jwt import create_access_token, create_refresh_token, decode_token
from auth.password import hash_password, verify_password
from config.settings import settings
from models.schemas import (
    InviteRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from observability.logger import get_logger
from storage.database import get_db
from storage.models import RefreshToken, Tenant, User

router = APIRouter()
logger = get_logger(__name__)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _issue_tokens(user: User, db: Session) -> TokenResponse:
    access = create_access_token(user.id, user.tenant_id, user.role)
    refresh, expires_at = create_refresh_token(user.id)

    db.add(RefreshToken(
        user_id=user.id,
        token_hash=_token_hash(refresh),
        expires_at=expires_at,
    ))
    db.commit()

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        business_role=user.business_role or "consultor",
    )


# ─── Register ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    if db.query(Tenant).filter(Tenant.slug == body.org_slug).first():
        raise HTTPException(status_code=409, detail="Organization slug already taken")

    tenant = Tenant(name=body.org_name, slug=body.org_slug)
    db.add(tenant)
    db.flush()   # get tenant.id before committing

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role="owner",
        business_role=body.business_role if body.business_role in ("consultor", "colaborador") else "consultor",
        tenant_id=tenant.id,
    )
    db.add(user)
    db.flush()

    tenant.owner_id = user.id
    db.commit()
    db.refresh(user)

    logger.info(f"New org registered: slug={body.org_slug} owner={body.email}")
    return _issue_tokens(user, db)


# ─── Login ───────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    return _issue_tokens(user, db)


# ─── Refresh ─────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
def refresh_tokens(body: RefreshRequest, db: Session = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Wrong token type")

    stored = db.query(RefreshToken).filter(
        RefreshToken.token_hash == _token_hash(body.refresh_token),
        RefreshToken.revoked == False,
    ).first()

    if not stored:
        raise HTTPException(status_code=401, detail="Token revoked or not found")

    expires = stored.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = db.query(User).filter(User.id == payload["sub"], User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Revoke old token (rotation)
    stored.revoked = True
    db.commit()

    return _issue_tokens(user, db)


# ─── Logout ──────────────────────────────────────────────────────────────────

@router.post("/logout")
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    stored = db.query(RefreshToken).filter(
        RefreshToken.token_hash == _token_hash(body.refresh_token),
    ).first()
    if stored:
        stored.revoked = True
        db.commit()
    return {"message": "Logged out"}


# ─── Me ──────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
def me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tenant = db.query(Tenant).filter(Tenant.id == current_user.tenant_id).first()
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        business_role=current_user.business_role or "consultor",
        tenant_id=current_user.tenant_id,
        tenant_name=tenant.name if tenant else "",
        tenant_slug=tenant.slug if tenant else "",
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


# ─── Invite ──────────────────────────────────────────────────────────────────

@router.post("/invite")
def invite(
    body: InviteRequest,
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can invite")

    from datetime import timedelta
    from jose import jwt as jose_jwt

    expire = datetime.now(timezone.utc) + timedelta(days=7)
    invite_payload = {
        "email": body.email,
        "tenant_id": current_user.tenant_id,
        "role": body.role,
        "exp": expire,
        "type": "invite",
    }
    invite_token = jose_jwt.encode(invite_payload, settings.secret_key, algorithm="HS256")

    return {
        "invite_token": invite_token,
        "expires_in": "7 days",
        "note": "In production, email this token to the invitee.",
    }


# ─── Accept invite ───────────────────────────────────────────────────────────

@router.post("/accept-invite", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def accept_invite(
    invite_token: str,
    password: str,
    full_name: str,
    db: Session = Depends(get_db),
):
    try:
        payload = decode_token(invite_token)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or expired invite token")

    if payload.get("type") != "invite":
        raise HTTPException(status_code=400, detail="Not an invite token")

    email = payload["email"]
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=email,
        hashed_password=hash_password(password),
        full_name=full_name,
        role=payload["role"],
        tenant_id=payload["tenant_id"],
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"Invite accepted: email={email} tenant={payload['tenant_id']}")
    return _issue_tokens(user, db)
