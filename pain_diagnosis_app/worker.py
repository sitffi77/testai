"""
Worker threads for asynchronous ML inference and document processing.

Provides QThread-based workers for:
- ML prediction (MLWorker)
- Document parsing and flag detection (DocumentProcessingWorker)
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from models.ml_engine import MLEngine
from nlp.models import DocumentText, FlagsSummary
from nlp.flag_detector import FlagDetector

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


class DocumentProcessingWorker(QThread):
    """
    Worker thread for document parsing and flag detection.
    
    Processes uploaded documents (PDF/DOCX) and detects clinical flags.
    
    Signals:
        finished: Emitted when processing is complete with (DocumentText, FlagsSummary).
        error: Emitted when an error occurs with error message.
        progress: Emitted during processing with status message.
    """
    
    # Signals
    finished = pyqtSignal(object, object)  # DocumentText, FlagsSummary
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(
        self,
        file_path: str,
        flag_detector: Optional[FlagDetector] = None,
        patient_age: Optional[int] = None,
        patient_temperature: Optional[float] = None
    ) -> None:
        """
        Initialize the document processing worker.
        
        Args:
            file_path: Path to the document file.
            flag_detector: FlagDetector instance (created if None).
            patient_age: Patient age (overrides extracted from document).
            patient_temperature: Patient temperature (overrides extracted).
        """
        super().__init__()
        self.file_path = file_path
        self.flag_detector = flag_detector or FlagDetector()
        self.patient_age = patient_age
        self.patient_temperature = patient_temperature
        self._is_running = False
    
    def run(self) -> None:
        """Execute document parsing and flag detection."""
        self._is_running = True
        
        try:
            from pathlib import Path
            
            # Step 1: Parse document
            self.progress.emit("📄 Парсинг документа...")
            logger.info(f"DocumentProcessingWorker: Parsing {self.file_path}")
            
            path = Path(self.file_path)
            suffix = path.suffix.lower()
            
            if suffix == '.pdf':
                from parsers.pdf_parser import PDFParser
                parser = PDFParser()
                doc_text = parser.parse(self.file_path)
            elif suffix in ['.docx', '.doc']:
                from parsers.docx_parser import DOCXParser
                parser = DOCXParser()
                doc_text = parser.parse(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            self.progress.emit("🔍 Анализ клинических флагов...")
            logger.info(f"DocumentProcessingWorker: Detecting flags in {len(doc_text.content)} chars")
            
            # Step 2: Detect flags
            flags_summary = self.flag_detector.detect_flags(
                doc_text,
                patient_age=self.patient_age,
                patient_temperature=self.patient_temperature
            )
            
            self.progress.emit("✅ Обработка завершена")
            logger.info(
                f"DocumentProcessingWorker: Completed - "
                f"🔴{len(flags_summary.red_flags)} ⚫{len(flags_summary.black_flags)} "
                f"🟡{len(flags_summary.yellow_flags)} 🔵{len(flags_summary.blue_flags)}"
            )
            
            # Emit results
            self.finished.emit(doc_text, flags_summary)
            
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            
        except ValueError as e:
            error_msg = f"Parse error: {e}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            
        except Exception as e:
            error_msg = f"Processing failed: {e}"
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
