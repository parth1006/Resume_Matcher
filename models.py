from sqlalchemy import Column, Integer, String, Float, JSON, Text
from sqlalchemy.orm import declarative_base
Base = declarative_base()

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    raw_text = Column(Text)
    skills = Column(JSON)
    experience_years = Column(Float)
    education = Column(JSON)
    embedding = Column(JSON)
    companies = Column(JSON, nullable=True)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    jd_text = Column(Text)
    required_skills = Column(JSON)
    nice_to_have_skills = Column(JSON)
    embedding = Column(JSON)
