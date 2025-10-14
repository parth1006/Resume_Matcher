from sqlalchemy import Column, Integer, String, Float, Text, TypeDecorator
from sqlalchemy.orm import declarative_base
import json

Base = declarative_base()

class JSONType(TypeDecorator):
    """Custom JSON type that works reliably with SQLite."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert Python object to JSON string for storage."""
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        """Convert JSON string back to Python object."""
        if value is None:
            return None
        return json.loads(value)


class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    raw_text = Column(Text)
    skills = Column(JSONType)  # ✅ Changed from JSON to JSONType
    experience_years = Column(Float)
    education = Column(JSONType)  # ✅ Changed from JSON to JSONType
    embedding = Column(JSONType)  # ✅ Changed from JSON to JSONType
    companies = Column(JSONType, nullable=True)  # ✅ Changed from JSON to JSONType
    resume_path = Column(String, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    jd_text = Column(Text)
    required_skills = Column(JSONType)  # ✅ Changed from JSON to JSONType
    nice_to_have_skills = Column(JSONType)  # ✅ Changed from JSON to JSONType
    embedding = Column(JSONType)  # ✅ Changed from JSON to JSONType