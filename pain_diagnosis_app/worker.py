"""
Worker thread for asynchronous ML inference.

Runs ML prediction in a separate thread to avoid blocking the GUI.
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from models.ml_engine import MLEngine

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
