"""
Main entry point for Pain Diagnosis Application.

Initializes logging, controller, and starts the PyQt6 application.
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE
from controller import get_controller
from ui.main_window import MainWindow


def setup_logging() -> None:
    """
    Configure application logging.

    Sets up file and console handlers with specified format and level.
    """
    # Create logs directory if needed
    log_file = Path(LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific levels for noisy libraries
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")


def create_application() -> QApplication:
    """
    Create and configure Qt application instance.

    Returns:
        Configured QApplication instance.
    """
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Pain Diagnosis System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Medical AI Lab")
    
    # Enable High DPI scaling
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Set default font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    return app


def initialize_controller(model_path: str = None) -> bool:
    """
    Initialize the application controller.

    Args:
        model_path: Optional path to ML model file.

    Returns:
        True if initialization successful, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing controller...")
    
    controller = get_controller()
    
    success = controller.initialize(model_path)
    
    if success:
        logger.info("Controller initialized successfully")
    else:
        logger.error("Controller initialization failed")
    
    return success


def main() -> int:
    """
    Main application entry point.

    Returns:
        Application exit code.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Pain Diagnosis Application")

    # Create Qt application
    app = create_application()

    # Initialize controller
    # Try to load model from default path or create demo model
    model_success = initialize_controller()
    
    if not model_success:
        logger.warning("Running with limited functionality (no model)")

    # Create main window
    window = MainWindow()
    
    # Connect controller signals to UI slots
    controller = get_controller(main_window=window)
    
    # When form submits data, run prediction
    window.diagnose_requested.connect(controller.run_prediction)
    
    # When prediction completes, display results
    controller.prediction_completed.connect(window.display_results)
    controller.prediction_completed.connect(lambda: window.set_loading_state(False))
    
    # When prediction starts, show loading state
    controller.prediction_started.connect(lambda: window.set_loading_state(True))
    
    # Handle errors
    controller.prediction_error.connect(
        lambda msg: window.show_error("Ошибка диагностики", msg)
    )
    controller.prediction_error.connect(lambda: window.set_loading_state(False))
    
    # Status messages
    controller.status_message.connect(window.set_status_message)
    
    # Save requested
    def on_save_requested():
        """Handle save request from UI."""
        input_form = window.input_form
        results_view = window.results_view
        
        if input_form and results_view and results_view.current_result:
            form_data = input_form.get_form_data()
            
            clinical_data = {
                "age": form_data.get("age"),
                "pain_intensity": form_data.get("pain_intensity"),
                "duration_days": form_data.get("duration_days"),
                "frequency_per_week": form_data.get("frequency_per_week"),
                "sleep_hours": form_data.get("sleep_hours"),
                "stress_level": form_data.get("stress_level"),
                "pain_location": form_data.get("pain_location"),
                "pain_type": form_data.get("pain_type"),
                "trigger_factor": form_data.get("trigger_factor"),
                "relief_factor": form_data.get("relief_factor"),
                "medication_use": form_data.get("medication_use"),
            }
            
            success = controller.save_diagnosis_to_db(
                patient_name=form_data.get("patient_name", "Unknown"),
                patient_gender=form_data.get("gender", ""),
                clinical_data=clinical_data,
                prediction_result=results_view.current_result,
            )
            
            if success:
                window.set_status_message("Результаты успешно сохранены")
            else:
                window.show_warning(
                    "Предупреждение",
                    "Не удалось сохранить результаты в базу данных"
                )
    
    window.save_requested.connect(on_save_requested)
    
    # Connect training tab signals
    def on_training_started():
        """Handle training start request from training tab."""
        controller.start_training()
    
    if hasattr(window, 'training_view'):
        window.training_view.start_training_signal.connect(on_training_started)

    # Show window
    window.show()
    logger.info("Application window displayed")

    # Run event loop
    exit_code = app.exec()
    
    # Cleanup
    controller.cleanup()
    logger.info("Application shutdown complete")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
