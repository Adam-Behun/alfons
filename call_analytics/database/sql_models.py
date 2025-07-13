from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

# Assuming SQLAlchemy is available; define Base for models
Base = declarative_base()

class Recording(Base):
    """
    SQL model for storing recording metadata and basic info.
    Used for structured analytics if PostgreSQL is integrated.
    """
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float)
    outcome = Column(String)  # e.g., 'success', 'failure'
    participants = Column(String)  # JSON string or comma-separated

    analytics = relationship("Analytic", back_populates="recording")

class Analytic(Base):
    """
    SQL model for analytics data derived from recordings.
    Includes metrics like success score, timing, etc.
    """
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=False)
    success_score = Column(Float)
    objection_count = Column(Integer)
    average_response_time = Column(Float)
    analysis_date = Column(DateTime, default=datetime.utcnow)

    recording = relationship("Recording", back_populates="analytics")
    patterns = relationship("Pattern", back_populates="analytic")

class Pattern(Base):
    """
    SQL model for extracted patterns from analytics.
    Stores learnings like scripts, objections, etc.
    """
    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True, index=True)
    analytic_id = Column(Integer, ForeignKey("analytics.id"), nullable=False)
    pattern_type = Column(String)  # e.g., 'objection', 'success_phrase'
    description = Column(String)
    frequency = Column(Integer)
    is_successful = Column(Boolean, default=False)

    analytic = relationship("Analytic", back_populates="patterns")

# Example usage (for testing; assumes a session/engine setup elsewhere)
if __name__ == "__main__":
    # This would require an engine and session to test, but here's a placeholder
    print("SQL models defined. To create tables, use: Base.metadata.create_all(engine)")