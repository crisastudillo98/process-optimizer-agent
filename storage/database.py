from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config.settings import settings

# ── Engine ──────────────────────────────────────────────────────────────────
# SQLite en dev → cambiar DATABASE_URL en .env para PostgreSQL en prod
# Ejemplo prod: postgresql+psycopg2://user:pass@host:5432/dbname
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)

# SQLite: habilitar WAL para concurrencia básica
if "sqlite" in settings.database_url:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """Dependency para FastAPI — cierra la sesión al terminar el request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
