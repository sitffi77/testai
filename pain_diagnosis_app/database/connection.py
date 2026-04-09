"""
Database connection module for Pain Diagnosis Application.

Provides database initialization, session management, and CRUD operations.
"""

import logging
from typing import Optional, Any, List, Dict
from datetime import datetime

from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from models.db_models import (
    Base,
    Patient,
    ClinicalData,
    Scale,
    Diagnosis,
    get_engine,
    get_session,
    init_db,
)
from config import DATABASE_URL

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for Pain Diagnosis Application.

    Handles all database operations including CRUD for patients,
    clinical data, and diagnoses.

    Attributes:
        engine: SQLAlchemy engine instance.
    """

    def __init__(self) -> None:
        """Initialize the database manager."""
        self.engine = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize database connection and create tables.

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            self.engine = get_engine()
            init_db(self.engine)
            self._initialized = True
            logger.info("Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._initialized = False
            return False

    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized

    def create_patient(
        self,
        full_name: str,
        date_of_birth: Optional[datetime] = None,
        gender: Optional[str] = None,
    ) -> Optional[Patient]:
        """
        Create a new patient record.

        Args:
            full_name: Patient's full name.
            date_of_birth: Date of birth.
            gender: Gender.

        Returns:
            Created Patient object or None on failure.
        """
        if not self._initialized:
            logger.error("Database not initialized")
            return None

        session = get_session(self.engine)
        try:
            patient = Patient(
                full_name=full_name,
                date_of_birth=date_of_birth,
                gender=gender,
            )
            session.add(patient)
            session.commit()
            session.refresh(patient)
            logger.info(f"Created patient: {patient.id}")
            return patient
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create patient: {e}")
            return None
        finally:
            session.close()

    def get_patient(self, patient_id: int) -> Optional[Patient]:
        """
        Get patient by ID.

        Args:
            patient_id: Patient ID.

        Returns:
            Patient object or None if not found.
        """
        if not self._initialized:
            return None

        session = get_session(self.engine)
        try:
            patient = session.query(Patient).filter(Patient.id == patient_id).first()
            return patient
        except SQLAlchemyError as e:
            logger.error(f"Failed to get patient: {e}")
            return None
        finally:
            session.close()

    def create_clinical_data(
        self,
        patient_id: int,
        age: Optional[int] = None,
        pain_intensity: Optional[float] = None,
        duration_days: Optional[int] = None,
        frequency_per_week: Optional[int] = None,
        sleep_hours: Optional[float] = None,
        stress_level: Optional[float] = None,
        pain_location: Optional[str] = None,
        pain_type: Optional[str] = None,
        trigger_factor: Optional[str] = None,
        relief_factor: Optional[str] = None,
        medication_use: Optional[str] = None,
    ) -> Optional[ClinicalData]:
        """
        Create a new clinical data record.

        Args:
            patient_id: Associated patient ID.
            age: Patient age.
            pain_intensity: Pain intensity score (0-10).
            duration_days: Duration in days.
            frequency_per_week: Frequency per week.
            sleep_hours: Sleep hours.
            stress_level: Stress level.
            pain_location: Pain location.
            pain_type: Pain type.
            trigger_factor: Trigger factor.
            relief_factor: Relief factor.
            medication_use: Medication use.

        Returns:
            Created ClinicalData object or None on failure.
        """
        if not self._initialized:
            return None

        session = get_session(self.engine)
        try:
            clinical_data = ClinicalData(
                patient_id=patient_id,
                age=age,
                pain_intensity=pain_intensity,
                duration_days=duration_days,
                frequency_per_week=frequency_per_week,
                sleep_hours=sleep_hours,
                stress_level=stress_level,
                pain_location=pain_location,
                pain_type=pain_type,
                trigger_factor=trigger_factor,
                relief_factor=relief_factor,
                medication_use=medication_use,
            )
            session.add(clinical_data)
            session.commit()
            session.refresh(clinical_data)
            logger.info(f"Created clinical data: {clinical_data.id}")
            return clinical_data
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create clinical data: {e}")
            return None
        finally:
            session.close()

    def create_diagnosis(
        self,
        patient_id: int,
        clinical_data_id: int,
        predicted_class: str,
        confidence: Optional[float] = None,
        probabilities: Optional[Dict[str, float]] = None,
        shap_values: Optional[Dict[str, float]] = None,
        interpretation: Optional[str] = None,
    ) -> Optional[Diagnosis]:
        """
        Create a new diagnosis record.

        Args:
            patient_id: Associated patient ID.
            clinical_data_id: Associated clinical data ID.
            predicted_class: Predicted pain class.
            confidence: Confidence score.
            probabilities: Class probabilities dict.
            shap_values: SHAP values dict.
            interpretation: Human-readable interpretation.

        Returns:
            Created Diagnosis object or None on failure.
        """
        if not self._initialized:
            return None

        session = get_session(self.engine)
        try:
            diagnosis = Diagnosis(
                patient_id=patient_id,
                clinical_data_id=clinical_data_id,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities_json=probabilities,
                shap_values_json=shap_values,
                interpretation=interpretation,
            )
            session.add(diagnosis)
            session.commit()
            session.refresh(diagnosis)
            logger.info(f"Created diagnosis: {diagnosis.id}")
            return diagnosis
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to create diagnosis: {e}")
            return None
        finally:
            session.close()

    def get_patient_diagnoses(self, patient_id: int) -> List[Diagnosis]:
        """
        Get all diagnoses for a patient.

        Args:
            patient_id: Patient ID.

        Returns:
            List of Diagnosis objects.
        """
        if not self._initialized:
            return []

        session = get_session(self.engine)
        try:
            diagnoses = (
                session.query(Diagnosis)
                .filter(Diagnosis.patient_id == patient_id)
                .order_by(Diagnosis.created_at.desc())
                .all()
            )
            return diagnoses
        except SQLAlchemyError as e:
            logger.error(f"Failed to get patient diagnoses: {e}")
            return []
        finally:
            session.close()

    def get_all_patients(self) -> List[Patient]:
        """
        Get all patients.

        Returns:
            List of Patient objects.
        """
        if not self._initialized:
            return []

        session = get_session(self.engine)
        try:
            patients = session.query(Patient).all()
            return patients
        except SQLAlchemyError as e:
            logger.error(f"Failed to get patients: {e}")
            return []
        finally:
            session.close()

    def save_diagnosis_result(
        self,
        patient_full_name: str,
        patient_gender: Optional[str],
        clinical_data_dict: Dict[str, Any],
        prediction_result: Dict[str, Any],
        interpretation: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Save complete diagnosis result including patient, clinical data, and diagnosis.

        Args:
            patient_full_name: Patient's full name.
            patient_gender: Patient's gender.
            clinical_data_dict: Dictionary with clinical data fields.
            prediction_result: ML prediction result dictionary.
            interpretation: Human-readable interpretation.

        Returns:
            Dictionary with created IDs or None on failure.
        """
        # Create or find patient
        patient = self.create_patient(
            full_name=patient_full_name,
            gender=patient_gender,
        )
        
        if patient is None:
            return None

        # Create clinical data
        clinical_data = self.create_clinical_data(
            patient_id=patient.id,
            age=clinical_data_dict.get("age"),
            pain_intensity=clinical_data_dict.get("pain_intensity"),
            duration_days=clinical_data_dict.get("duration_days"),
            frequency_per_week=clinical_data_dict.get("frequency_per_week"),
            sleep_hours=clinical_data_dict.get("sleep_hours"),
            stress_level=clinical_data_dict.get("stress_level"),
            pain_location=clinical_data_dict.get("pain_location"),
            pain_type=clinical_data_dict.get("pain_type"),
            trigger_factor=clinical_data_dict.get("trigger_factor"),
            relief_factor=clinical_data_dict.get("relief_factor"),
            medication_use=clinical_data_dict.get("medication_use"),
        )

        if clinical_data is None:
            return None

        # Create diagnosis
        diagnosis = self.create_diagnosis(
            patient_id=patient.id,
            clinical_data_id=clinical_data.id,
            predicted_class=prediction_result.get("class", ""),
            confidence=prediction_result.get("probabilities", {}).get(
                prediction_result.get("class", ""), 0.0
            ),
            probabilities=prediction_result.get("probabilities"),
            shap_values=prediction_result.get("shap_values"),
            interpretation=interpretation,
        )

        if diagnosis is None:
            return None

        return {
            "patient_id": patient.id,
            "clinical_data_id": clinical_data.id,
            "diagnosis_id": diagnosis.id,
        }


# Singleton instance
_db_manager_instance: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get or create the singleton database manager instance.

    Returns:
        DatabaseManager instance.
    """
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance
