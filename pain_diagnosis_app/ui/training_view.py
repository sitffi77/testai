"""
Training view UI component for Pain Diagnosis Application.

Provides interface for managing model fine-tuning, displaying statistics,
logs, and training progress.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTextEdit, QProgressBar, QGroupBox, 
    QMessageBox, QFormLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingView(QWidget):
    """
    Interface for managing model fine-tuning.
    
    Displays training statistics, logs, and allows starting the training process.
    
    Signals:
        start_training_signal: Emitted when user clicks the train button.
    """
    
    start_training_signal = pyqtSignal()
    
    def __init__(self):
        """Initialize the training view UI."""
        super().__init__()
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Statistics group
        stats_group = QGroupBox("📊 Статистика обучающей выборки")
        stats_layout = QFormLayout()
        
        self.lbl_samples = QLabel("0")
        self.lbl_samples.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        
        self.lbl_classes = QLabel("-")
        self.lbl_classes.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.lbl_last_acc = QLabel("-")
        self.lbl_last_acc.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        
        stats_layout.addRow("Доступно примеров:", self.lbl_samples)
        stats_layout.addRow("Классов в данных:", self.lbl_classes)
        stats_layout.addRow("Последняя точность:", self.lbl_last_acc)
        
        stats_group.setLayout(stats_layout)
        
        # Control group
        ctrl_group = QGroupBox("⚙️ Управление обучением")
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(10)
        
        # Info label
        info_label = QLabel(
            "Нажмите кнопку ниже для запуска дообучения модели на новых данных.\n"
            "Процесс может занять несколько минут."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        ctrl_layout.addWidget(info_label)
        
        # Train button
        self.btn_train = QPushButton("🚀 Запустить дообучение")
        self.btn_train.clicked.connect(self._on_start_training)
        self.btn_train.setStyleSheet("""
            QPushButton {
                font-weight: bold; 
                font-size: 14px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        ctrl_layout.addWidget(self.btn_train)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        ctrl_layout.addWidget(self.progress)
        
        # Log section
        log_label = QLabel("📋 Журнал процесса:")
        log_label.setStyleSheet("font-weight: bold;")
        ctrl_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFontFamily("Consolas")
        self.log_text.setFontPointSize(9)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        ctrl_layout.addWidget(self.log_text)
        
        ctrl_group.setLayout(ctrl_layout)
        
        # Chart group
        chart_group = QGroupBox("📈 Метрики качества")
        chart_layout = QVBoxLayout()
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: white;")
        
        chart_layout.addWidget(self.canvas)
        chart_group.setLayout(chart_layout)
        
        # Add all groups to main layout
        layout.addWidget(stats_group)
        layout.addWidget(ctrl_group)
        layout.addWidget(chart_group, stretch=1)
        
        self.setLayout(layout)
        
        # Initialize chart
        self._init_chart()
        
    def _init_chart(self):
        """Initialize the metrics chart."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'График метрик появится после обучения', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='#999')
        ax.axis('off')
        self.canvas.draw()
        
    def update_stats(self, count: int, classes: int = 0, acc: float = None):
        """
        Update statistics display.
        
        Args:
            count: Number of training samples.
            classes: Number of classes in the data.
            acc: Last accuracy value (0-1 scale).
        """
        self.lbl_samples.setText(str(count))
        self.lbl_classes.setText(str(classes) if classes > 0 else "-")
        
        if acc is not None:
            self.lbl_last_acc.setText(f"{acc:.1%}")
            if acc >= 0.75:
                self.lbl_last_acc.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
            elif acc >= 0.5:
                self.lbl_last_acc.setStyleSheet("font-weight: bold; font-size: 14px; color: #FF9800;")
            else:
                self.lbl_last_acc.setStyleSheet("font-weight: bold; font-size: 14px; color: #F44336;")
        else:
            self.lbl_last_acc.setText("-")
        
    def log_message(self, message: str):
        """
        Append a message to the log.
        
        Args:
            message: Message to append.
        """
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _on_start_training(self):
        """Handle train button click."""
        self.btn_train.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate mode
        self.log_message("Запрос на обучение отправлен...")
        self.start_training_signal.emit()
        
    def training_finished(self, success: bool, message: str):
        """
        Handle training completion.
        
        Args:
            success: Whether training was successful.
            message: Result message.
        """
        self.btn_train.setEnabled(True)
        self.progress.setVisible(False)
        
        if success:
            self.log_message(f"✅ Успешно: {message}")
            QMessageBox.information(self, "Обучение завершено", message)
        else:
            self.log_message(f"❌ Ошибка: {message}")
            QMessageBox.critical(self, "Ошибка обучения", message)
            
    def plot_metrics(self, history: dict):
        """
        Plot training metrics from history.
        
        Args:
            history: Dictionary containing 'loss' and optionally 'val_loss' lists.
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Plot loss history if available
        if 'loss' in history and history['loss']:
            epochs = range(len(history['loss']))
            ax.plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
            
            if 'val_loss' in history and history['val_loss']:
                ax.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
            
            ax.legend(loc='upper right')
            ax.set_title("Динамика потерь во время обучения")
            ax.set_xlabel("Эпоха / Итерация")
            ax.set_ylabel("Loss (mlogloss)")
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            self.canvas.draw()


# Import for timestamp
from PyQt6.QtCore import QDateTime
