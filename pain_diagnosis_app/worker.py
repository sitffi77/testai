"""
Worker threads for asynchronous operations in Pain Diagnosis Application.

Includes ML inference worker and model training worker.
"""

import logging
import traceback
from typing import Dict, Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot

from models.ml_engine import MLEngine
from models.ml_trainer import MLTrainer
from database.connection import DatabaseManager
from config import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


class MLWorker(QThread):
    """
    Worker thread for running ML inference asynchronously.

    Signals:
        finished: Emitted when prediction is complete with result dict.
        error: Emitted when an error occurs with error message.

    Attributes:
        ml_engine: ML engine instance for prediction.
        input_data: Input data dictionary for prediction.
    """

    # Signals
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ml_engine: MLEngine, input_data: Dict[str, Any]) -> None:
        """
        Initialize the ML worker.

        Args:
            ml_engine: ML engine instance to use for prediction.
            input_data: Dictionary containing clinical data for prediction.
        """
        super().__init__()
        self.ml_engine = ml_engine
        self.input_data = input_data
        self._is_running = False

    def run(self) -> None:
        """
        Execute the ML prediction in this thread.

        This method is called when start() is invoked on the thread.
        It performs the prediction and emits signals with results or errors.
        """
        self._is_running = True
        
        try:
            logger.info("MLWorker: Starting prediction...")
            
            # Perform prediction
            result = self.ml_engine.predict(self.input_data)
            
            # Generate interpretation
            interpretation = self.ml_engine.get_interpretation(result)
            result["interpretation"] = interpretation
            
            logger.info(f"MLWorker: Prediction completed - {result.get('class', 'unknown')}")
            
            # Emit result
            self.finished.emit(result)
            
        except RuntimeError as e:
            error_msg = f"Model not loaded: {e}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            logger.error(error_msg)
            self.error.emit(error_msg)
        
        finally:
            self._is_running = False

    def stop(self) -> None:
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.wait()

    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._is_running


class TrainingWorker(QThread):
    """
    Worker thread for running model fine-tuning asynchronously.

    Signals:
        finished: Emitted when training completes (success, message).
        log_update: Emitted with log messages during training.

    Attributes:
        model_path: Path to save the trained model.
        trainer: MLTrainer instance.
    """

    # Signals
    finished = pyqtSignal(bool, str)  # success, message
    log_update = pyqtSignal(str)

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the training worker.

        Args:
            model_path: Path to save the trained model. Uses default if None.
        """
        super().__init__()
        self.model_path = model_path if model_path else str(DEFAULT_MODEL_PATH)
        self.trainer = MLTrainer()
        self._is_running = False

    @pyqtSlot()
    def run(self) -> None:
        """Execute the training process in this thread."""
        self._is_running = True
        
        try:
            self.log_update.emit("Подключение к базе данных...")
            db_manager = DatabaseManager()
            db_success = db_manager.initialize()
            
            if not db_success:
                raise RuntimeError("Не удалось подключиться к базе данных")
            
            session = db_manager.get_session()
            
            self.log_update.emit("Сбор и подготовка данных...")
            
            # Define progress callback
            def progress_callback(message: str):
                self.log_update.emit(message)
            
            # Run training pipeline
            metrics = self.trainer.run_full_pipeline(
                session, 
                self.model_path,
                progress_callback=progress_callback
            )
            
            session.close()
            
            # Build success message
            msg = (
                f"✅ Обучение завершено успешно!\n\n"
                f"📊 Примеров использовано: {metrics['samples_count']}\n"
                f"📈 Точность на валидации: {metrics['accuracy']:.1%}\n"
                f"📁 Классов: {metrics['classes']}\n"
                f"💾 Модель сохранена: {metrics['model_path']}\n"
                f"{'✅ Порог качества пройден' if metrics.get('meets_threshold') else '⚠️ Точность ниже порога'}"
            )
            
            self.finished.emit(True, msg)
            
        except ValueError as e:
            # Not enough samples
            error_msg = f"Недостаточно данных для обучения: {str(e)}"
            self.log_update.emit(f"❌ {error_msg}")
            self.finished.emit(False, error_msg)
            
        except Exception as e:
            error_msg = f"Ошибка обучения: {str(e)}"
            self.log_update.emit(f"❌ {error_msg}")
            self.log_update.emit(traceback.format_exc())
            self.finished.emit(False, str(e))
        
        finally:
            self._is_running = False

    def stop(self) -> None:
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.wait()

    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._is_running
