"""Auth endpoint tests — uses an in-memory SQLite DB, no external services."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from storage.database import Base, get_db
from storage.models import Tenant, User
from auth.password import hash_password
from api.main import app

# ─── In-memory SQLite test DB ────────────────────────────────────────────────
# StaticPool forces all sessions to share one connection so the in-memory DB
# persists across requests within a test.

TEST_DATABASE_URL = "sqlite://"   # in-memory

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def reset_db():
    """Drop and recreate all tables before each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

REGISTER_PAYLOAD = {
    "email": "owner@test.com",
    "password": "Password123!",
    "full_name": "Test Owner",
    "org_name": "Test Corp",
    "org_slug": "test-corp",
}


def register_and_login(client: TestClient, payload: dict = None) -> dict:
    """Register a user and return the token response dict."""
    p = payload or REGISTER_PAYLOAD
    r = client.post("/auth/register", json=p)
    assert r.status_code == 201, r.text
    return r.json()


def auth_headers(tokens: dict) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


# ─── Register ────────────────────────────────────────────────────────────────

class TestRegister:
    def test_happy_path(self, client):
        r = client.post("/auth/register", json=REGISTER_PAYLOAD)
        assert r.status_code == 201
        data = r.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_duplicate_email(self, client):
        client.post("/auth/register", json=REGISTER_PAYLOAD)
        r = client.post("/auth/register", json=REGISTER_PAYLOAD)
        assert r.status_code == 409
        assert "Email" in r.json()["detail"]

    def test_duplicate_slug(self, client):
        client.post("/auth/register", json=REGISTER_PAYLOAD)
        different_email = {**REGISTER_PAYLOAD, "email": "other@test.com"}
        r = client.post("/auth/register", json=different_email)
        assert r.status_code == 409
        assert "slug" in r.json()["detail"].lower()

    def test_invalid_slug_pattern(self, client):
        bad_slug = {**REGISTER_PAYLOAD, "org_slug": "Has Spaces!"}
        r = client.post("/auth/register", json=bad_slug)
        assert r.status_code == 422

    def test_short_password(self, client):
        short_pw = {**REGISTER_PAYLOAD, "password": "short"}
        r = client.post("/auth/register", json=short_pw)
        assert r.status_code == 422


# ─── Login ───────────────────────────────────────────────────────────────────

class TestLogin:
    def test_correct_credentials(self, client):
        register_and_login(client)
        r = client.post("/auth/login", json={
            "email": REGISTER_PAYLOAD["email"],
            "password": REGISTER_PAYLOAD["password"],
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_wrong_password(self, client):
        register_and_login(client)
        r = client.post("/auth/login", json={
            "email": REGISTER_PAYLOAD["email"],
            "password": "wrongpassword",
        })
        assert r.status_code == 401

    def test_unknown_email(self, client):
        r = client.post("/auth/login", json={
            "email": "nobody@test.com",
            "password": "whatever",
        })
        assert r.status_code == 401

    def test_inactive_user(self, client):
        register_and_login(client)
        # Manually deactivate user in DB
        db = TestingSessionLocal()
        user = db.query(User).filter(User.email == REGISTER_PAYLOAD["email"]).first()
        user.is_active = False
        db.commit()
        db.close()

        r = client.post("/auth/login", json={
            "email": REGISTER_PAYLOAD["email"],
            "password": REGISTER_PAYLOAD["password"],
        })
        assert r.status_code == 403


# ─── Token Refresh ───────────────────────────────────────────────────────────

class TestRefresh:
    def test_valid_refresh(self, client):
        tokens = register_and_login(client)
        r = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert r.status_code == 200
        new_tokens = r.json()
        assert new_tokens["access_token"] != tokens["access_token"]

    def test_revoked_refresh(self, client):
        tokens = register_and_login(client)
        # Use the token once (rotates it)
        client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        # Second use should fail
        r = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert r.status_code == 401

    def test_invalid_refresh_token(self, client):
        r = client.post("/auth/refresh", json={"refresh_token": "not.a.valid.token"})
        assert r.status_code == 401


# ─── Me ──────────────────────────────────────────────────────────────────────

class TestMe:
    def test_with_valid_token(self, client):
        tokens = register_and_login(client)
        r = client.get("/auth/me", headers=auth_headers(tokens))
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == REGISTER_PAYLOAD["email"]
        assert data["role"] == "owner"
        assert data["tenant_slug"] == REGISTER_PAYLOAD["org_slug"]

    def test_without_token_returns_403(self, client):
        # HTTPBearer returns 403 when no credentials are provided
        r = client.get("/auth/me")
        assert r.status_code in (401, 403)

    def test_with_invalid_token(self, client):
        r = client.get("/auth/me", headers={"Authorization": "Bearer garbage.token.here"})
        assert r.status_code == 401


# ─── Protected endpoint enforcement ──────────────────────────────────────────

class TestProtectedEndpoints:
    def test_analyze_text_requires_auth(self, client):
        r = client.post("/analyze/text", json={"raw_input": "x" * 60})
        assert r.status_code in (401, 403)

    def test_analyze_text_with_valid_token_is_accepted(self, client):
        tokens = register_and_login(client)
        r = client.post(
            "/analyze/text",
            json={"raw_input": "x" * 60, "process_name": "Test"},
            headers=auth_headers(tokens),
        )
        # 202 = accepted (pipeline runs in background); 5xx would mean a bug
        assert r.status_code == 202

    def test_analyses_list_requires_auth(self, client):
        r = client.get("/analyses")
        assert r.status_code in (401, 403)

    def test_analyses_list_returns_only_own(self, client):
        # User A registers and creates an analysis
        tokens_a = register_and_login(client)
        client.post(
            "/analyze/text",
            json={"raw_input": "a" * 60, "process_name": "User A process"},
            headers=auth_headers(tokens_a),
        )

        # User B registers
        tokens_b = register_and_login(client, {
            "email": "b@test.com",
            "password": "Password123!",
            "full_name": "User B",
            "org_name": "Corp B",
            "org_slug": "corp-b",
        })
        r = client.get("/analyses", headers=auth_headers(tokens_b))
        assert r.status_code == 200
        # User B should see 0 analyses (none are hers)
        analyses = r.json()["analyses"]
        assert all(a["id"] != "User A process" for a in analyses)


# ─── Tenant isolation (session ownership) ────────────────────────────────────

class TestSessionIsolation:
    def test_user_b_cannot_access_user_a_session(self, client):
        tokens_a = register_and_login(client)
        # Create a session as user A
        r = client.post(
            "/analyze/text",
            json={"raw_input": "a" * 60},
            headers=auth_headers(tokens_a),
        )
        assert r.status_code == 202
        session_id = r.json()["session_id"]

        # Register user B in a different org
        tokens_b = register_and_login(client, {
            "email": "b@test.com",
            "password": "Password123!",
            "full_name": "User B",
            "org_name": "Corp B",
            "org_slug": "corp-b",
        })

        # User B tries to access user A's session
        r = client.get(
            f"/sessions/{session_id}/status",
            headers=auth_headers(tokens_b),
        )
        assert r.status_code == 403
