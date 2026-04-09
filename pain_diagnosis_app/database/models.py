"""Database models for Pain Diagnosis Application."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, 
    Text, Boolean, JSON, create_engine
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

Base = declarative_base()


class Patient(Base):
    """Модель пациента."""
    
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(200), nullable=False, index=True)
    gender = Column(String(10), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'full_name': self.full_name,
            'gender': self.gender,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Diagnosis(Base):
    """Модель диагноза/результата диагностики."""
    
    __tablename__ = 'diagnoses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    
    # Clinical data
    age = Column(Integer, nullable=True)
    pain_intensity = Column(Float, nullable=True)
    pain_location = Column(String(100), nullable=True)
    pain_type = Column(String(100), nullable=True)
    duration_days = Column(Integer, nullable=True)
    
    # ML prediction results
    predicted_class = Column(String(50), nullable=True)
    prediction_probability = Column(Float, nullable=True)
    shap_plot_base64 = Column(Text, nullable=True)
    interpretation = Column(Text, nullable=True)
    
    # Flag summary
    red_flags_count = Column(Integer, default=0)
    yellow_flags_count = Column(Integer, default=0)
    blue_flags_count = Column(Integer, default=0)
    black_flags_count = Column(Integer, default=0)
    total_severity_score = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")
    flags = relationship("Flag", back_populates="diagnosis", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'age': self.age,
            'pain_intensity': self.pain_intensity,
            'pain_location': self.pain_location,
            'pain_type': self.pain_type,
            'duration_days': self.duration_days,
            'predicted_class': self.predicted_class,
            'prediction_probability': self.prediction_probability,
            'interpretation': self.interpretation,
            'red_flags_count': self.red_flags_count,
            'yellow_flags_count': self.yellow_flags_count,
            'blue_flags_count': self.blue_flags_count,
            'black_flags_count': self.black_flags_count,
            'total_severity_score': self.total_severity_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Flag(Base):
    """Модель клинического флага."""
    
    __tablename__ = 'flags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    diagnosis_id = Column(Integer, ForeignKey('diagnoses.id'), nullable=False)
    
    # Flag details
    flag_type = Column(String(20), nullable=False, index=True)  # red, yellow, blue, black
    flag_id = Column(String(10), nullable=False)  # R01, Y05, etc.
    flag_name = Column(String(200), nullable=True)
    
    # Detection details
    confidence = Column(String(10), nullable=True)  # High, Medium, Low
    context_snippet = Column(Text, nullable=True)
    keyword_matched = Column(String(100), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    diagnosis = relationship("Diagnosis", back_populates="flags")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'diagnosis_id': self.diagnosis_id,
            'flag_type': self.flag_type,
            'flag_id': self.flag_id,
            'flag_name': self.flag_name,
            'confidence': self.confidence,
            'context_snippet': self.context_snippet,
            'keyword_matched': self.keyword_matched,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ClinicalData(Base):
    """Модель для хранения полных клинических данных (JSON)."""
    
    __tablename__ = 'clinical_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    diagnosis_id = Column(Integer, ForeignKey('diagnoses.id'), nullable=False, unique=True)
    
    # Store all clinical data as JSON
    data_json = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    diagnosis = relationship("Diagnosis", backref="clinical_data_record")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'diagnosis_id': self.diagnosis_id,
            'data': self.data_json,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


def init_db(engine_url: str) -> tuple:
    """
    Инициализация базы данных.
    
    Args:
        engine_url: URL подключения к БД.
        
    Returns:
        Кортеж (engine, Session).
    """
    engine = create_engine(engine_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def get_sample_flag_data() -> List[Dict[str, Any]]:
    """
    Получить пример данных флагов для тестирования.
    
    Returns:
        Список словарей с данными флагов.
    """
    return [
        {
            'flag_type': 'red',
            'flag_id': 'R04',
            'flag_name': 'Возраст дебюта боли <20 или >55 лет',
            'confidence': 'High',
            'context_snippet': 'Пациент 62 года, боль в спине',
            'is_active': True
        },
        {
            'flag_type': 'red',
            'flag_id': 'R17',
            'flag_name': 'Температура >37°C',
            'confidence': 'High',
            'context_snippet': 'Температура тела 37.5°C',
            'is_active': True
        },
        {
            'flag_type': 'yellow',
            'flag_id': 'Y02',
            'flag_name': 'Депрессия',
            'confidence': 'Medium',
            'context_snippet': 'Отмечается депрессивное состояние',
            'is_active': True
        },
        {
            'flag_type': 'yellow',
            'flag_id': 'Y05',
            'flag_name': 'Кинезиофобия',
            'confidence': 'Medium',
            'context_snippet': 'Страх движения, кинезиофобия',
            'is_active': True
        },
        {
            'flag_type': 'blue',
            'flag_id': 'B05',
            'flag_name': 'Низкая удовлетворённость работой',
            'confidence': 'Low',
            'context_snippet': 'Неудовлетворённость работой',
            'is_active': True
        }
    ]
