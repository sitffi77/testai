"""
Input form widget for clinical data entry.

Provides form fields for all clinical features with validation.
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


class ClinicalInputForm(QWidget):
    """
    Form widget for entering clinical data.

    Provides input fields for all numerical and categorical features
    required for ML prediction.

    Signals:
        submit_clicked: Emitted when form is submitted with validated data.
        reset_clicked: Emitted when form is reset.

    Attributes:
        fields: Dictionary mapping field names to input widgets.
    """

    submit_clicked = pyqtSignal(dict)
    reset_clicked = pyqtSignal()

    # Predefined options for categorical fields
    GENDER_OPTIONS = ["M", "F", "Other"]
    PAIN_LOCATION_OPTIONS = [
        "head", "neck", "back", "chest",
        "abdomen", "limbs", "generalized"
    ]
    PAIN_TYPE_OPTIONS = [
        "aching", "sharp", "burning", "throbbing",
        "shooting", "dull", "stabbing"
    ]
    TRIGGER_FACTOR_OPTIONS = [
        "movement", "stress", "weather", "food",
        "none", "unknown"
    ]
    RELIEF_FACTOR_OPTIONS = [
        "rest", "medication", "heat", "cold",
        "movement", "none"
    ]
    MEDICATION_USE_OPTIONS = [
        "none", "nsaids", "opioids", "antidepressants",
        "anticonvulsants", "muscle_relaxants", "combination"
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the clinical input form.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.fields: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Patient info group
        patient_group = self._create_patient_info_group()
        main_layout.addWidget(patient_group)

        # Numerical features group
        numerical_group = self._create_numerical_features_group()
        main_layout.addWidget(numerical_group)

        # Categorical features group
        categorical_group = self._create_categorical_features_group()
        main_layout.addWidget(categorical_group)

        # Buttons
        button_layout = self._create_button_layout()
        main_layout.addLayout(button_layout)

    def _create_patient_info_group(self) -> QGroupBox:
        """Create patient information group box."""
        group = QGroupBox("Информация о пациенте")
        layout = QFormLayout()
        layout.setSpacing(8)

        # Patient name
        self.fields["patient_name"] = QLineEdit()
        self.fields["patient_name"].setPlaceholderText("Иванов Иван Иванович")
        layout.addRow("ФИО пациента:", self.fields["patient_name"])

        # Gender
        self.fields["gender"] = QComboBox()
        self.fields["gender"].addItems(self.GENDER_OPTIONS)
        layout.addRow("Пол:", self.fields["gender"])

        # Age
        self.fields["age"] = QSpinBox()
        self.fields["age"].setRange(1, 120)
        self.fields["age"].setValue(40)
        layout.addRow("Возраст:", self.fields["age"])

        group.setLayout(layout)
        return group

    def _create_numerical_features_group(self) -> QGroupBox:
        """Create numerical features group box."""
        group = QGroupBox("Числовые показатели")
        layout = QFormLayout()
        layout.setSpacing(8)

        # Pain intensity (0-10)
        self.fields["pain_intensity"] = QDoubleSpinBox()
        self.fields["pain_intensity"].setRange(0.0, 10.0)
        self.fields["pain_intensity"].setDecimals(1)
        self.fields["pain_intensity"].setValue(5.0)
        self.fields["pain_intensity"].setSingleStep(0.5)
        layout.addRow("Интенсивность боли (0-10):", self.fields["pain_intensity"])

        # Duration in days
        self.fields["duration_days"] = QSpinBox()
        self.fields["duration_days"].setRange(0, 36500)
        self.fields["duration_days"].setValue(30)
        layout.addRow("Длительность (дни):", self.fields["duration_days"])

        # Frequency per week
        self.fields["frequency_per_week"] = QSpinBox()
        self.fields["frequency_per_week"].setRange(0, 168)
        self.fields["frequency_per_week"].setValue(7)
        layout.addRow("Частота в неделю:", self.fields["frequency_per_week"])

        # Sleep hours
        self.fields["sleep_hours"] = QDoubleSpinBox()
        self.fields["sleep_hours"].setRange(0.0, 24.0)
        self.fields["sleep_hours"].setDecimals(1)
        self.fields["sleep_hours"].setValue(7.0)
        self.fields["sleep_hours"].setSingleStep(0.5)
        layout.addRow("Часы сна:", self.fields["sleep_hours"])

        # Stress level (0-10)
        self.fields["stress_level"] = QDoubleSpinBox()
        self.fields["stress_level"].setRange(0.0, 10.0)
        self.fields["stress_level"].setDecimals(1)
        self.fields["stress_level"].setValue(5.0)
        self.fields["stress_level"].setSingleStep(0.5)
        layout.addRow("Уровень стресса (0-10):", self.fields["stress_level"])

        group.setLayout(layout)
        return group

    def _create_categorical_features_group(self) -> QGroupBox:
        """Create categorical features group box."""
        group = QGroupBox("Категориальные признаки")
        layout = QFormLayout()
        layout.setSpacing(8)

        # Pain location
        self.fields["pain_location"] = QComboBox()
        self.fields["pain_location"].addItems(self.PAIN_LOCATION_OPTIONS)
        layout.addRow("Локализация боли:", self.fields["pain_location"])

        # Pain type
        self.fields["pain_type"] = QComboBox()
        self.fields["pain_type"].addItems(self.PAIN_TYPE_OPTIONS)
        layout.addRow("Тип боли:", self.fields["pain_type"])

        # Trigger factor
        self.fields["trigger_factor"] = QComboBox()
        self.fields["trigger_factor"].addItems(self.TRIGGER_FACTOR_OPTIONS)
        layout.addRow("Провоцирующий фактор:", self.fields["trigger_factor"])

        # Relief factor
        self.fields["relief_factor"] = QComboBox()
        self.fields["relief_factor"].addItems(self.RELIEF_FACTOR_OPTIONS)
        layout.addRow("Облегчающий фактор:", self.fields["relief_factor"])

        # Medication use
        self.fields["medication_use"] = QComboBox()
        self.fields["medication_use"].addItems(self.MEDICATION_USE_OPTIONS)
        layout.addRow("Прием лекарств:", self.fields["medication_use"])

        group.setLayout(layout)
        return group

    def _create_button_layout(self) -> QHBoxLayout:
        """Create button layout."""
        layout = QHBoxLayout()
        layout.setSpacing(10)

        # Submit button
        self.submit_btn = QPushButton("Диагностировать")
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.submit_btn.clicked.connect(self._on_submit)
        layout.addWidget(self.submit_btn)

        # Reset button
        self.reset_btn = QPushButton("Сбросить")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        layout.addStretch()

        return layout

    def _on_submit(self) -> None:
        """Handle submit button click."""
        data = self.get_form_data()
        
        # Basic validation
        if not data.get("patient_name"):
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Поле 'ФИО пациента' обязательно для заполнения"
            )
            return
        
        if not data.get("gender"):
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Поле 'Пол' обязательно для заполнения"
            )
            return

        self.submit_clicked.emit(data)

    def _on_reset(self) -> None:
        """Handle reset button click."""
        self.reset_form()
        self.reset_clicked.emit()

    def get_form_data(self) -> Dict[str, Any]:
        """
        Get current form data as dictionary.

        Returns:
            Dictionary containing all form field values.
        """
        data = {}

        # Get text fields
        data["patient_name"] = self.fields["patient_name"].text().strip()
        data["gender"] = self.fields["gender"].currentText()

        # Get numerical fields
        data["age"] = self.fields["age"].value()
        data["pain_intensity"] = self.fields["pain_intensity"].value()
        data["duration_days"] = self.fields["duration_days"].value()
        data["frequency_per_week"] = self.fields["frequency_per_week"].value()
        data["sleep_hours"] = self.fields["sleep_hours"].value()
        data["stress_level"] = self.fields["stress_level"].value()

        # Get categorical fields
        data["pain_location"] = self.fields["pain_location"].currentText()
        data["pain_type"] = self.fields["pain_type"].currentText()
        data["trigger_factor"] = self.fields["trigger_factor"].currentText()
        data["relief_factor"] = self.fields["relief_factor"].currentText()
        data["medication_use"] = self.fields["medication_use"].currentText()

        return data

    def reset_form(self) -> None:
        """Reset all form fields to default values."""
        # Clear text fields
        self.fields["patient_name"].clear()

        # Reset spin boxes to defaults
        self.fields["age"].setValue(40)
        self.fields["pain_intensity"].setValue(5.0)
        self.fields["duration_days"].setValue(30)
        self.fields["frequency_per_week"].setValue(7)
        self.fields["sleep_hours"].setValue(7.0)
        self.fields["stress_level"].setValue(5.0)

        # Reset combo boxes to first item
        for field_name in ["gender", "pain_location", "pain_type",
                          "trigger_factor", "relief_factor", "medication_use"]:
            self.fields[field_name].setCurrentIndex(0)

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable all form fields.

        Args:
            enabled: True to enable, False to disable.
        """
        for widget in self.fields.values():
            widget.setEnabled(enabled)
        self.submit_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
