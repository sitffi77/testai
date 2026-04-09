"""
SQLAlchemy database models for Pain Diagnosis Application.

Defines ORM models for patients, clinical_data, scales, diagnoses, and training_data tables.
Uses SQLAlchemy 2.0 declarative style.
"""

from datetime import datetime
from typing import Optional, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Boolean,
    create_engine,
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

from config import DATABASE_URL

Base = declarative_base()


class Patient(Base):
    """
    Patient model representing the patients table.

    Attributes:
        id: Primary key.
        full_name: Patient's full name.
        date_of_birth: Date of birth.
        gender: Gender (M/F/Other).
        created_at: Record creation timestamp.
    """

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    clinical_records = relationship("ClinicalData", back_populates="patient", cascade="all, delete-orphan")
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    training_samples = relationship("TrainingData", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, full_name='{self.full_name}')>"


class ClinicalData(Base):
    """
    Clinical data model representing the clinical_data table.

    Stores raw clinical measurements and observations.

    Attributes:
        id: Primary key.
        patient_id: Foreign key to patients table.
        age: Patient age at time of examination.
        pain_intensity: Pain intensity score (0-10).
        duration_days: Duration of pain in days.
        frequency_per_week: How often pain occurs per week.
        sleep_hours: Average sleep hours per day.
        stress_level: Stress level score (0-10).
        pain_location: Anatomical location of pain.
        pain_type: Type/character of pain.
        trigger_factor: Factors that trigger pain.
        relief_factor: Factors that relieve pain.
        medication_use: Current medication usage.
        created_at: Record creation timestamp.
    """

    __tablename__ = "clinical_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Numerical features
    age = Column(Integer, nullable=True)
    pain_intensity = Column(Float, nullable=True)
    duration_days = Column(Integer, nullable=True)
    frequency_per_week = Column(Integer, nullable=True)
    sleep_hours = Column(Float, nullable=True)
    stress_level = Column(Float, nullable=True)
    
    # Categorical features
    pain_location = Column(String(100), nullable=True)
    pain_type = Column(String(100), nullable=True)
    trigger_factor = Column(String(255), nullable=True)
    relief_factor = Column(String(255), nullable=True)
    medication_use = Column(String(255), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="clinical_records")
    diagnosis = relationship("Diagnosis", back_populates="clinical_data", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ClinicalData(id={self.id}, patient_id={self.patient_id})>"


class Scale(Base):
    """
    Scale model representing the scales table.

    Stores reference data for pain assessment scales.

    Attributes:
        id: Primary key.
        name: Scale name (e.g., VAS, NRS).
        description: Scale description.
        min_value: Minimum scale value.
        max_value: Maximum scale value.
    """

    __tablename__ = "scales"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    min_value = Column(Float, default=0.0)
    max_value = Column(Float, default=10.0)

    def __repr__(self) -> str:
        return f"<Scale(id={self.id}, name='{self.name}')>"


class Diagnosis(Base):
    """
    Diagnosis model representing the diagnoses table.

    Stores ML classification results with probabilities and SHAP values.

    Attributes:
        id: Primary key.
        patient_id: Foreign key to patients table.
        clinical_data_id: Foreign key to clinical_data table.
        predicted_class: ML-predicted pain type.
        confidence: Confidence score for prediction.
        probabilities_json: JSON dict of class probabilities.
        shap_values_json: JSON dict of SHAP feature contributions.
        interpretation: Human-readable interpretation.
        is_verified: Whether the diagnosis was verified by a doctor.
        verified_by: Name of the doctor who verified.
        created_at: Record creation timestamp.
    """

    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    clinical_data_id = Column(Integer, ForeignKey("clinical_data.id"), nullable=False)
    
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=True)
    probabilities_json = Column(JSON, nullable=True)
    shap_values_json = Column(JSON, nullable=True)
    interpretation = Column(Text, nullable=True)
    is_verified = Column(Boolean, default=False)
    verified_by = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="diagnoses")
    clinical_data = relationship("ClinicalData", back_populates="diagnosis")

    def __repr__(self) -> str:
        return f"<Diagnosis(id={self.id}, predicted_class='{self.predicted_class}')>"


class TrainingData(Base):
    """
    Training data model representing the training_data table.
    
    Stores labeled data for model fine-tuning.
    Populated when a doctor confirms or corrects a diagnosis.

    Attributes:
        id: Primary key.
        patient_id: Foreign key to patients table.
        clinical_data_id: Foreign key to clinical_data table (optional).
        features_json: JSON serialized feature vector after preprocessing.
        true_label: The correct diagnosis label according to the doctor.
        source: Source of the data ('manual', 'import', 'correction', 'verification').
        comment: Optional comment from the doctor.
        created_at: Record creation timestamp.
    """

    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    clinical_data_id = Column(Integer, ForeignKey("clinical_data.id"), nullable=True)
    
    # Input features (serialized JSON feature vector after preprocessing)
    features_json = Column(JSON, nullable=False)
    
    # Target label (true diagnosis according to doctor)
    true_label = Column(String(50), nullable=False)
    
    # Metadata
    source = Column(String(20), default='manual')  # 'manual', 'import', 'correction', 'verification'
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="training_samples")

    def __repr__(self) -> str:
        return f"<TrainingData(id={self.id}, patient_id={self.patient_id}, true_label='{self.true_label}')>"


class FlagResultModel(Base):
    """
    Flag result model representing the flags table.
    
    Stores red/yellow/blue/black flag检测结果 for patient safety.
    """

    __tablename__ = "flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    flag_type = Column(String(20), nullable=False)  # red, yellow, blue, black
    flag_id = Column(Integer, nullable=True)
    confidence = Column(String(10), nullable=True)
    context_snippet = Column(Text, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='active')

    def __repr__(self) -> str:
        return f"<FlagResultModel(id={self.id}, flag_type='{self.flag_type}')>"


def get_engine() -> Any:
    """
    Create and return a SQLAlchemy engine.

    Returns:
        SQLAlchemy engine instance.

    Raises:
        Exception: If connection fails.
    """
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            echo=False,
        )
        return engine
    except Exception as e:
        raise ConnectionError(f"Failed to create database engine: {e}")


def get_session(engine: Optional[Any] = None) -> Any:
    """
    Create and return a SQLAlchemy session.

    Args:
        engine: Optional SQLAlchemy engine. If None, creates new one.

    Returns:
        SQLAlchemy session instance.
    """
    if engine is None:
        engine = get_engine()
    
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()


def init_db(engine: Optional[Any] = None) -> None:
    """
    Initialize database by creating all tables.

    Args:
        engine: Optional SQLAlchemy engine. If None, creates new one.
    """
    if engine is None:
        engine = get_engine()
    
    Base.metadata.create_all(bind=engine)
