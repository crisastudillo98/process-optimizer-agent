from uuid import uuid4
from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.sql import func
from storage.database import Base



class Analysis(Base):
    """Tabla principal de análisis persistidos."""
    __tablename__ = "analyses"

    id              = Column(String, primary_key=True, index=True)   # session_id del pipeline
    process_name    = Column(String(255), nullable=False, default="Sin nombre")
    status          = Column(String(50), default="running")          # running | completed | error
    raw_input       = Column(Text, nullable=True)

    result_json     = Column(Text, nullable=True)

    score           = Column(Float, nullable=True)
    cycle_time_reduction_pct = Column(Float, nullable=True)
    automation_coverage_pct  = Column(Float, nullable=True)

    # BPMN file paths — persisted so they survive server restarts
    bpmn_tobe_path  = Column(String, nullable=True)
    bpmn_asis_path  = Column(String, nullable=True)

    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    completed_at    = Column(DateTime(timezone=True), nullable=True)
    has_errors      = Column(Boolean, default=False)

    # Auth — nullable for backwards compat with pre-auth rows
    user_id         = Column(String, ForeignKey("users.id"), nullable=True)
    tenant_id       = Column(String, ForeignKey("tenants.id"), nullable=True)

    def __repr__(self):
        return f"<Analysis id={self.id} name={self.process_name} status={self.status}>"


class ChatMessage(Base):
    """Mensajes de chat persistidos por sesión."""
    __tablename__ = "chat_messages"

    id         = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("analyses.id"), nullable=False, index=True)
    role       = Column(String(50), nullable=False)   # user | assistant
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ChatMessage session={self.session_id} role={self.role}>"


class Tenant(Base):
    __tablename__ = "tenants"

    id         = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name       = Column(String(255), nullable=False)
    slug       = Column(String(100), unique=True, nullable=False, index=True)
    plan       = Column(String(50), default="free")   # free | pro | enterprise
    created_at = Column(DateTime, server_default=func.now())
    owner_id   = Column(String, nullable=True)        # back-filled after user creation

    def __repr__(self):
        return f"<Tenant slug={self.slug} plan={self.plan}>"


class User(Base):
    __tablename__ = "users"

    id              = Column(String, primary_key=True, default=lambda: str(uuid4()))
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name       = Column(String(255), nullable=False)
    role            = Column(String(50), default="member")      # owner|admin|member|viewer
    business_role   = Column(String(50), default="consultor")   # consultor|colaborador
    tenant_id       = Column(String, ForeignKey("tenants.id"), nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, server_default=func.now())
    last_login      = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User email={self.email} role={self.role}>"


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked    = Column(Boolean, default=False)

    def __repr__(self):
        return f"<RefreshToken user_id={self.user_id} revoked={self.revoked}>"


class ProcessCollaborator(Base):
    """Links users to specific processes as collaborators."""
    __tablename__ = "process_collaborators"

    id          = Column(String, primary_key=True, default=lambda: str(uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False, index=True)
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    invited_by  = Column(String, ForeignKey("users.id"), nullable=False)
    status      = Column(String(50), default="pending")   # pending|active|completed
    created_at  = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<ProcessCollaborator analysis={self.analysis_id} user={self.user_id} status={self.status}>"


class Invitation(Base):
    """Email invitations to collaborate on a process."""
    __tablename__ = "invitations"

    id          = Column(String, primary_key=True, default=lambda: str(uuid4()))
    email       = Column(String(255), nullable=False)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    invited_by  = Column(String, ForeignKey("users.id"), nullable=False)
    role        = Column(String(50), default="colaborador")
    token       = Column(String, unique=True, nullable=False)
    status      = Column(String(50), default="pending")   # pending|accepted|expired
    expires_at  = Column(DateTime, nullable=False)
    created_at  = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Invitation email={self.email} analysis={self.analysis_id} status={self.status}>"


class Notification(Base):
    """In-app notifications for users."""
    __tablename__ = "notifications"

    id         = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    type       = Column(String(100), nullable=False)
    title      = Column(String(255), nullable=False)
    message    = Column(Text, nullable=False)
    data       = Column(Text, nullable=True)   # JSON with extra context
    read       = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Notification user={self.user_id} type={self.type} read={self.read}>"
