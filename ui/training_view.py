# ui/training_view.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QProgressBar, QGroupBox, 
                             QMessageBox, QFormLayout, QLineEdit, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io

class TrainingView(QWidget):
    """
    Интерфейс для управления дообучением модели.
    Отображает статистику, логи и позволяет запустить процесс.
    """
    start_training_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Статус и статистика
        stats_group = QGroupBox("Статистика обучающей выборки")
        stats_layout = QFormLayout()
        self.lbl_samples = QLabel("0")
        self.lbl_classes = QLabel("-")
        self.lbl_last_acc = QLabel("-")
        stats_layout.addRow("Доступно примеров:", self.lbl_samples)
        stats_layout.addRow("Классов:", self.lbl_classes)
        stats_layout.addRow("Последняя точность:", self.lbl_last_acc)
        stats_group.setLayout(stats_layout)
        
        # Управление
        ctrl_group = QGroupBox("Управление обучением")
        ctrl_layout = QVBoxLayout()
        
        self.btn_train = QPushButton("🚀 Запустить дообучение")
        self.btn_train.clicked.connect(self.on_start_training)
        self.btn_train.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFontFamily("Consolas")
        self.log_text.setFontPointSize(9)
        
        ctrl_layout.addWidget(self.btn_train)
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addWidget(QLabel("Журнал процесса:"))
        ctrl_layout.addWidget(self.log_text)
        
        ctrl_group.setLayout(ctrl_layout)
        
        # График
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        layout.addWidget(stats_group)
        layout.addWidget(ctrl_group)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_stats(self, count: int, classes: int, acc: float):
        self.lbl_samples.setText(str(count))
        self.lbl_classes.setText(str(classes))
        self.lbl_last_acc.setText(f"{acc:.2%}" if acc else "-")
        
    def log_message(self, message: str):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def on_start_training(self):
        self.btn_train.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.log_message("Запрос на обучение отправлен...")
        self.start_training_signal.emit()
        
    def training_finished(self, success: bool, message: str):
        self.btn_train.setEnabled(True)
        self.progress.setVisible(False)
        if success:
            self.log_message(f"✅ Успешно: {message}")
            QMessageBox.information(self, "Готово", message)
        else:
            self.log_message(f"❌ Ошибка: {message}")
            QMessageBox.critical(self, "Ошибка", message)
            
    def plot_metrics(self, history: dict):
        """Отрисовка графика потерь (если есть история)"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Пример отрисовки, если передана история лоссов
        if 'loss' in history:
            ax.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                ax.plot(history['val_loss'], label='Val Loss')
            ax.legend()
            ax.set_title("Динамика потерь")
            ax.set_xlabel("Эпоха")
            ax.set_ylabel("Loss")
            self.canvas.draw()
