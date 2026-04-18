from sqlalchemy import Column, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func
from storage.database import Base

class Analysis(Base):
    """Tabla principal de análisis persistidos."""
    __tablename__ = "analyses"

    id              = Column(String, primary_key=True, index=True)   # session_id del pipeline
    process_name    = Column(String(255), nullable=False, default="Sin nombre")
    status          = Column(String(50), default="running")          # running | completed | error
    raw_input       = Column(Text, nullable=True)

    # Resultado serializado como JSON
    result_json     = Column(Text, nullable=True)   # full report al completar

    # Métricas rápidas para listar sin deserializar el JSON completo
    score           = Column(Float, nullable=True)   # waste_percentage como score de optimización
    cycle_time_reduction_pct = Column(Float, nullable=True)
    automation_coverage_pct  = Column(Float, nullable=True)

    # Auditoría
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    completed_at    = Column(DateTime(timezone=True), nullable=True)
    has_errors      = Column(Boolean, default=False)

    def __repr__(self):
        return f"<Analysis id={self.id} name={self.process_name} status={self.status}>"
