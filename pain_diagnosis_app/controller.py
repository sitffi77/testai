"""
Controller module for Pain Diagnosis Application.

Coordinates between UI, ML engine, and database.
Implements business logic and manages application state.
"""

import logging
from typing import Dict, Any, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from models.ml_engine import get_ml_engine, MLEngine
from database.connection import get_db_manager, DatabaseManager
from worker import MLWorker
from config import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


class Controller(QObject):
    """
    Main controller for the Pain Diagnosis Application.

    Coordinates UI interactions, ML inference, and database operations.
    Follows MVC pattern where controller handles business logic.

    Signals:
        prediction_started: Emitted when prediction begins.
        prediction_completed: Emitted when prediction finishes with result dict.
        prediction_error: Emitted when prediction fails with error message.
        db_operation_completed: Emitted when DB operation completes.
        status_message: Emitted for status updates.

    Attributes:
        ml_engine: ML engine instance.
        db_manager: Database manager instance.
        current_worker: Current ML worker thread if running.
    """

    # Signals for UI updates
    prediction_started = pyqtSignal()
    prediction_completed = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    db_operation_completed = pyqtSignal(dict)
    status_message = pyqtSignal(str)

    def __init__(self) -> None:
        """Initialize the controller with ML engine and database manager."""
        super().__init__()
        
        self.ml_engine: MLEngine = get_ml_engine()
        self.db_manager: DatabaseManager = get_db_manager()
        
        self.current_worker: Optional[MLWorker] = None
        self._is_initialized = False
        
        logger.info("Controller initialized")

    def initialize(self, model_path: Optional[str] = None) -> bool:
        """
        Initialize the controller by loading ML model and connecting to database.

        Args:
            model_path: Optional path to ML model file. Uses default if None.

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Load ML model
            path = model_path if model_path else str(DEFAULT_MODEL_PATH)
            logger.info(f"Loading ML model from {path}")
            
            if not self.ml_engine.load_model(path):
                logger.error("Failed to load ML model")
                self.status_message.emit("Ошибка загрузки модели ML")
                return False
            
            # Initialize database (non-critical, app can work offline)
            logger.info("Initializing database connection")
            db_success = self.db_manager.initialize()
            
            if db_success:
                self.status_message.emit("Приложение готово к работе")
            else:
                self.status_message.emit("База данных недоступна. Работа без сохранения.")
            
            self._is_initialized = True
            logger.info("Controller initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Controller initialization failed: {e}")
            self.status_message.emit(f"Ошибка инициализации: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._is_initialized

    def validate_input_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate input clinical data.

        Args:
            data: Dictionary containing clinical data fields.

        Returns:
            Tuple of (is_valid, error_message).
        """
        errors = []

        # Validate numerical fields
        numerical_ranges = {
            "age": (0, 120, "Возраст"),
            "pain_intensity": (0, 10, "Интенсивность боли"),
            "duration_days": (0, 36500, "Длительность (дни)"),
            "frequency_per_week": (0, 168, "Частота в неделю"),
            "sleep_hours": (0, 24, "Часы сна"),
            "stress_level": (0, 10, "Уровень стресса"),
        }

        for field, (min_val, max_val, display_name) in numerical_ranges.items():
            value = data.get(field)
            if value is not None:
                try:
                    num_value = float(value)
                    if num_value < min_val or num_value > max_val:
                        errors.append(
                            f"{display_name} должен быть от {min_val} до {max_val}"
                        )
                except (ValueError, TypeError):
                    errors.append(f"{display_name} должен быть числом")

        # Validate required categorical fields
        required_categorical = [
            "gender",
            "pain_location",
            "pain_type",
        ]

        for field in required_categorical:
            value = data.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                errors.append(f"Поле '{field}' обязательно для заполнения")

        if errors:
            return False, "\n".join(errors)
        
        return True, ""

    def run_prediction(self, clinical_data: Dict[str, Any]) -> None:
        """
        Run ML prediction asynchronously.

        Args:
            clinical_data: Dictionary containing clinical data for prediction.
        """
        if not self._is_initialized:
            self.prediction_error.emit("Контроллер не инициализирован")
            return

        # Validate input
        is_valid, error_msg = self.validate_input_data(clinical_data)
        if not is_valid:
            self.prediction_error.emit(error_msg)
            return

        # Stop any existing worker
        if self.current_worker and self.current_worker.is_running():
            self.current_worker.stop()

        # Create and start new worker
        self.current_worker = MLWorker(self.ml_engine, clinical_data)
        self.current_worker.finished.connect(self._on_prediction_finished)
        self.current_worker.error.connect(self._on_prediction_error)
        
        self.prediction_started.emit()
        self.status_message.emit("Выполняется диагностика...")
        
        self.current_worker.start()
        logger.info("Prediction started in worker thread")

    def _on_prediction_finished(self, result: Dict[str, Any]) -> None:
        """
        Handle prediction completion.

        Args:
            result: Prediction result dictionary.
        """
        logger.info("Prediction finished successfully")
        self.prediction_completed.emit(result)
        self.status_message.emit("Диагностика завершена")

    def _on_prediction_error(self, error_msg: str) -> None:
        """
        Handle prediction error.

        Args:
            error_msg: Error message string.
        """
        logger.error(f"Prediction error: {error_msg}")
        self.prediction_error.emit(error_msg)
        self.status_message.emit(f"Ошибка диагностики: {error_msg}")

    def save_diagnosis_to_db(
        self,
        patient_name: str,
        patient_gender: str,
        clinical_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
    ) -> bool:
        """
        Save diagnosis results to database.

        Args:
            patient_name: Patient's full name.
            patient_gender: Patient's gender.
            clinical_data: Clinical data dictionary.
            prediction_result: ML prediction result dictionary.

        Returns:
            True if save successful, False otherwise.
        """
        if not self.db_manager.is_initialized():
            logger.warning("Database not initialized, cannot save")
            self.status_message.emit("База данных не подключена. Сохранение невозможно.")
            return False

        try:
            interpretation = prediction_result.get("interpretation", "")
            
            result = self.db_manager.save_diagnosis_result(
                patient_full_name=patient_name,
                patient_gender=patient_gender,
                clinical_data_dict=clinical_data,
                prediction_result=prediction_result,
                interpretation=interpretation,
            )

            if result:
                logger.info(f"Diagnosis saved: {result}")
                self.db_operation_completed.emit(result)
                self.status_message.emit("Результаты сохранены в базу данных")
                return True
            else:
                logger.error("Failed to save diagnosis")
                self.status_message.emit("Ошибка при сохранении результатов")
                return False

        except Exception as e:
            logger.error(f"Save to DB failed: {e}")
            self.status_message.emit(f"Ошибка БД: {e}")
            return False

    def get_shap_image(self, shap_plot_base64: str) -> Optional[QImage]:
        """
        Convert base64 SHAP plot to QImage for display.

        Args:
            shap_plot_base64: Base64-encoded PNG image string.

        Returns:
            QImage object or None on failure.
        """
        try:
            import base64
            from io import BytesIO
            
            img_data = base64.b64decode(shap_plot_base64)
            qimage = QImage()
            qimage.loadFromData(BytesIO(img_data).read())
            
            return qimage
            
        except Exception as e:
            logger.error(f"Failed to convert SHAP plot: {e}")
            return None

    def get_patient_history(self, patient_id: int) -> list:
        """
        Get diagnosis history for a patient.

        Args:
            patient_id: Patient ID.

        Returns:
            List of diagnosis records.
        """
        if not self.db_manager.is_initialized():
            return []
        
        return self.db_manager.get_patient_diagnoses(patient_id)

    def get_all_patients(self) -> list:
        """
        Get all patients from database.

        Returns:
            List of patient records.
        """
        if not self.db_manager.is_initialized():
            return []
        
        return self.db_manager.get_all_patients()

    def cleanup(self) -> None:
        """Clean up resources before application exit."""
        if self.current_worker and self.current_worker.is_running():
            self.current_worker.stop()
        
        logger.info("Controller cleanup completed")


# Singleton instance
_controller_instance: Optional[Controller] = None


def get_controller() -> Controller:
    """
    Get or create the singleton controller instance.

    Returns:
        Controller instance.
    """
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = Controller()
    return _controller_instance
