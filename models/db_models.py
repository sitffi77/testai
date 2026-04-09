# models/db_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    birth_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    
    clinical_records = relationship("ClinicalData", back_populates="patient")
    training_samples = relationship("TrainingData", back_populates="patient")

class ClinicalData(Base):
    __tablename__ = 'clinical_data'
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    pain_location = Column(String(50))
    pain_intensity = Column(Integer) # 0-10
    duration_days = Column(Integer)
    # ... остальные поля ...
    created_at = Column(DateTime, default=datetime.now)
    
    patient = relationship("Patient", back_populates="clinical_records")
    diagnosis = relationship("Diagnosis", uselist=False, back_populates="clinical_data")

class Diagnosis(Base):
    __tablename__ = 'diagnoses'
    id = Column(Integer, primary_key=True)
    clinical_data_id = Column(Integer, ForeignKey('clinical_data.id'))
    predicted_class = Column(String(50))
    probabilities_json = Column(JSON)
    shap_values_json = Column(JSON)
    is_verified = Column(Boolean, default=False) # Подтвержден ли диагноз врачом
    verified_by = Column(String(50), nullable=True)
    
    clinical_data = relationship("ClinicalData", back_populates="diagnosis")

# НОВАЯ ТАБЛИЦА ДЛЯ ОБУЧЕНИЯ
class TrainingData(Base):
    """
    Хранит размеченные данные для дообучения модели.
    Заполняется, когда врач подтверждает или исправляет диагноз.
    """
    __tablename__ = 'training_data'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    clinical_data_id = Column(Integer, ForeignKey('clinical_data.id'), nullable=True)
    
    # Входные признаки (сериализованный JSON вектор признаков после предобработки)
    features_json = Column(JSON, nullable=False)
    
    # Целевая метка (истинный диагноз по мнению врача)
    true_label = Column(String(50), nullable=False)
    
    # Метаданные
    source = Column(String(20), default='manual') # 'manual', 'import', 'correction'
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    patient = relationship("Patient", back_populates="training_samples")

class FlagResultModel(Base):
    __tablename__ = 'flags'
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    flag_type = Column(String(20)) # red, yellow, blue, black
    flag_id = Column(Integer)
    confidence = Column(String(10))
    context_snippet = Column(Text)
    detected_at = Column(DateTime, default=datetime.now)
    status = Column(String(20), default='active')
