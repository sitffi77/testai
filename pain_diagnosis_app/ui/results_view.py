"""
Results view widget for displaying ML prediction results.

Shows predicted class, probabilities, SHAP values, and interpretation.
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QScrollArea,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt6.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)


class ResultsView(QWidget):
    """
    Widget for displaying ML prediction results.

    Shows:
        - Predicted pain class with confidence
        - Class probabilities as progress bars
        - SHAP feature importance plot
        - Human-readable interpretation

    Signals:
        save_requested: Emitted when user requests to save results.

    Attributes:
        current_result: Current prediction result dictionary.
    """

    save_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the results view.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.current_result: Optional[Dict[str, Any]] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Prediction summary group
        summary_group = self._create_summary_group()
        content_layout.addWidget(summary_group)

        # Probabilities group
        prob_group = self._create_probabilities_group()
        content_layout.addWidget(prob_group)

        # SHAP plot group
        shap_group = self._create_shap_group()
        content_layout.addWidget(shap_group)

        # Interpretation group
        interp_group = self._create_interpretation_group()
        content_layout.addWidget(interp_group)

        # Save button
        save_btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Сохранить в базу данных")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        save_btn_layout.addWidget(self.save_btn)
        save_btn_layout.addStretch()
        content_layout.addLayout(save_btn_layout)

        content_layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Show placeholder
        self._show_placeholder()

    def _create_summary_group(self) -> QGroupBox:
        """Create prediction summary group box."""
        group = QGroupBox("Результат диагностики")
        layout = QVBoxLayout()

        # Predicted class label
        self.prediction_label = QLabel("Нет данных")
        self.prediction_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction_label)

        # Confidence label
        self.confidence_label = QLabel("Уверенность: --")
        self.confidence_label.setFont(QFont("Arial", 12))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)

        group.setLayout(layout)
        return group

    def _create_probabilities_group(self) -> QGroupBox:
        """Create probabilities group box."""
        group = QGroupBox("Вероятности классов")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Progress bars for each class
        self.prob_bars: Dict[str, QProgressBar] = {}
        self.prob_labels: Dict[str, QLabel] = {}

        class_names = {
            "nociceptive": "Ноцицептивная",
            "neuropathic": "Нейропатическая",
            "nociplastic": "Ноципластическая",
            "mixed": "Смешанная",
        }

        for class_key, class_name in class_names.items():
            row_layout = QHBoxLayout()

            # Label
            label = QLabel(class_name)
            label.setMinimumWidth(150)
            row_layout.addWidget(label)

            # Progress bar
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setMinimumWidth(200)
            row_layout.addWidget(bar)

            # Percentage label
            pct_label = QLabel("0.0%")
            pct_label.setMinimumWidth(60)
            row_layout.addWidget(pct_label)

            layout.addLayout(row_layout)

            self.prob_bars[class_key] = bar
            self.prob_labels[class_key] = pct_label

        group.setLayout(layout)
        return group

    def _create_shap_group(self) -> QGroupBox:
        """Create SHAP plot group box."""
        group = QGroupBox("Важность признаков (SHAP)")
        layout = QVBoxLayout()

        # Image label for SHAP plot
        self.shap_image_label = QLabel()
        self.shap_image_label.setMinimumHeight(300)
        self.shap_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shap_image_label.setStyleSheet(
            "QLabel { background-color: #f5f5f5; border: 1px dashed #ccc; }"
        )
        self.shap_image_label.setText("График будет отображен после диагностики")
        layout.addWidget(self.shap_image_label)

        group.setLayout(layout)
        return group

    def _create_interpretation_group(self) -> QGroupBox:
        """Create interpretation group box."""
        group = QGroupBox("Интерпретация результатов")
        layout = QVBoxLayout()

        # Text edit for interpretation
        self.interpretation_text = QTextEdit()
        self.interpretation_text.setReadOnly(True)
        self.interpretation_text.setMinimumHeight(150)
        self.interpretation_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.interpretation_text)

        group.setLayout(layout)
        return group

    def _show_placeholder(self) -> None:
        """Show placeholder state."""
        self.prediction_label.setText("Нет данных")
        self.prediction_label.setStyleSheet("color: #666;")
        self.confidence_label.setText("Уверенность: --")

        # Reset probability bars
        for bar in self.prob_bars.values():
            bar.setValue(0)
        for label in self.prob_labels.values():
            label.setText("0.0%")

        # Clear SHAP image
        self.shap_image_label.clear()
        self.shap_image_label.setText("График будет отображен после диагностики")
        self.shap_image_label.setStyleSheet(
            "QLabel { background-color: #f5f5f5; border: 1px dashed #ccc; color: #666; }"
        )

        # Clear interpretation
        self.interpretation_text.clear()
        self.interpretation_text.setPlaceholderText(
            "Интерпретация результатов появится здесь после диагностики"
        )

        # Disable save button
        self.save_btn.setEnabled(False)
        self.current_result = None

    def display_results(self, result: Dict[str, Any]) -> None:
        """
        Display prediction results.

        Args:
            result: Dictionary containing prediction results with keys:
                - class: Predicted class name
                - probabilities: Dict of class probabilities
                - shap_values: Dict of SHAP values
                - shap_plot_base64: Base64-encoded SHAP plot
                - interpretation: Human-readable interpretation
        """
        try:
            self.current_result = result

            # Update prediction label
            predicted_class = result.get("class", "unknown")
            class_translations = {
                "nociceptive": "Ноцицептивная (тканевая) боль",
                "neuropathic": "Нейропатическая боль",
                "nociplastic": "Ноципластическая боль",
                "mixed": "Смешанная боль",
            }
            display_class = class_translations.get(predicted_class, predicted_class)

            self.prediction_label.setText(display_class)
            self.prediction_label.setStyleSheet("color: #2E7D32;")

            # Update confidence
            probabilities = result.get("probabilities", {})
            confidence = probabilities.get(predicted_class, 0.0)
            confidence_pct = confidence * 100
            self.confidence_label.setText(f"Уверенность: {confidence_pct:.1f}%")

            # Update probability bars
            for class_key in self.prob_bars.keys():
                prob = probabilities.get(class_key, 0.0)
                prob_pct = prob * 100
                self.prob_bars[class_key].setValue(int(prob_pct))
                self.prob_labels[class_key].setText(f"{prob_pct:.1f}%")

                # Color code based on probability
                if prob_pct > 50:
                    color = "#4CAF50"  # Green
                elif prob_pct > 25:
                    color = "#FF9800"  # Orange
                else:
                    color = "#9E9E9E"  # Gray

                self.prob_bars[class_key].setStyleSheet(f"""
                    QProgressBar::chunk {{
                        background-color: {color};
                    }}
                """)

            # Update SHAP plot
            shap_base64 = result.get("shap_plot_base64", "")
            if shap_base64:
                self._display_shap_plot(shap_base64)

            # Update interpretation
            interpretation = result.get("interpretation", "")
            self.interpretation_text.setText(interpretation)

            # Enable save button
            self.save_btn.setEnabled(True)

            logger.info("Results displayed successfully")

        except Exception as e:
            logger.error(f"Failed to display results: {e}")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось отобразить результаты: {e}"
            )

    def _display_shap_plot(self, base64_data: str) -> None:
        """
        Display SHAP plot from base64 data.

        Args:
            base64_data: Base64-encoded PNG image string.
        """
        try:
            import base64
            from io import BytesIO

            img_data = base64.b64decode(base64_data)
            pixmap = QPixmap()
            pixmap.loadFromData(BytesIO(img_data).read())

            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.shap_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.shap_image_label.setPixmap(scaled_pixmap)
            self.shap_image_label.setStyleSheet("")
            self.shap_image_label.setMinimumHeight(300)

        except Exception as e:
            logger.error(f"Failed to display SHAP plot: {e}")
            self.shap_image_label.setText("Ошибка загрузки графика SHAP")

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        if self.current_result:
            self.save_requested.emit()
        else:
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Нет результатов для сохранения"
            )

    def clear_results(self) -> None:
        """Clear all displayed results."""
        self._show_placeholder()

    def set_loading_state(self, loading: bool) -> None:
        """
        Set loading state for the view.

        Args:
            loading: True if loading, False otherwise.
        """
        if loading:
            self.prediction_label.setText("Выполняется диагностика...")
            self.prediction_label.setStyleSheet("color: #1976D2;")
            self.save_btn.setEnabled(False)
            
            # Disable other interactions
            self.interpretation_text.clear()
            self.interpretation_text.setPlaceholderText("Загрузка...")
        else:
            self.save_btn.setEnabled(self.current_result is not None)

    def resizeEvent(self, event) -> None:
        """Handle resize event to update SHAP plot scaling."""
        super().resizeEvent(event)
        
        # Rescale SHAP plot if exists
        if self.current_result and self.shap_image_label.pixmap():
            shap_base64 = self.current_result.get("shap_plot_base64", "")
            if shap_base64:
                self._display_shap_plot(shap_base64)
