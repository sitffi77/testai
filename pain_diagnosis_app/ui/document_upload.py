"""Document upload widget for PDF and DOCX files."""

import logging
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QFont

logger = logging.getLogger(__name__)


class DocumentUploadWidget(QWidget):
    """Виджет загрузки медицинских документов.
    
    Поддерживает:
    - Выбор файлов через диалог (PDF, DOCX)
    - Drag-and-drop (опционально)
    - Отображение прогресса обработки
    - Валидацию формата файла
    
    Signals:
        file_selected: Сигнал при выборе файла (path: str).
        upload_started: Сигнал начала обработки.
        upload_finished: Сигнал завершения обработки.
    """
    
    file_selected = pyqtSignal(str)
    upload_started = pyqtSignal()
    upload_finished = pyqtSignal(object)  # DocumentText
    
    SUPPORTED_FORMATS = ["*.pdf", "*.docx", "*.doc"]
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Инициализация виджета загрузки.
        
        Args:
            parent: Родительский виджет.
        """
        super().__init__(parent)
        
        self._selected_file: Optional[str] = None
        self._setup_ui()
        
        logger.info("DocumentUploadWidget инициализирован")
    
    def _setup_ui(self) -> None:
        """Настройка пользовательского интерфейса."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Заголовок
        title_label = QLabel("📄 Загрузка медицинской выписки")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Фрейм для области загрузки
        self.drop_frame = QFrame()
        self.drop_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.drop_frame.setMinimumHeight(120)
        
        frame_layout = QVBoxLayout(self.drop_frame)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Иконка и текст
        self.icon_label = QLabel("📋")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.icon_label)
        
        self.info_label = QLabel(
            "Перетащите файл сюда или нажмите кнопку выбора\n"
            f"Поддерживаемые форматы: {', '.join(self.SUPPORTED_FORMATS)}"
        )
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #666;")
        frame_layout.addWidget(self.info_label)
        
        layout.addWidget(self.drop_frame)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.select_button = QPushButton("📁 Выбрать файл")
        self.select_button.setMinimumHeight(40)
        self.select_button.clicked.connect(self._on_select_clicked)
        buttons_layout.addWidget(self.select_button)
        
        self.clear_button = QPushButton("🗑️ Очистить")
        self.clear_button.setMinimumHeight(40)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.clear_button.setEnabled(False)
        buttons_layout.addWidget(self.clear_button)
        
        layout.addLayout(buttons_layout)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)
        
        # Метка статуса
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Стили для drag-and-drop
        self.drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
            QFrame:hover {
                border-color: #007bff;
                background-color: #eef6ff;
            }
        """)
        
        # Включаем drag-and-drop
        self.drop_frame.setAcceptDrops(True)
        self.drop_frame.dragEnterEvent = self._on_drag_enter
        self.drop_frame.dragLeaveEvent = self._on_drag_leave
        self.drop_frame.dropEvent = self._on_drop
    
    def _on_select_clicked(self) -> None:
        """Обработчик нажатия кнопки выбора файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите медицинский документ",
            "",
            f"Медицинские документы ({' '.join(self.SUPPORTED_FORMATS)});;Все файлы (*.*)"
        )
        
        if file_path:
            self._handle_file_selection(file_path)
    
    def _handle_file_selection(self, file_path: str) -> None:
        """Обработка выбранного файла.
        
        Args:
            file_path: Путь к выбранному файлу.
        """
        # Валидация расширения
        path = Path(file_path)
        if path.suffix.lower() not in ['.pdf', '.docx', '.doc']:
            self.status_label.setText(f"❌ Неверный формат: {path.suffix}")
            self.status_label.setStyleSheet("color: #dc3545;")
            logger.warning(f"Неверный формат файла: {file_path}")
            return
        
        self._selected_file = file_path
        
        # Обновление UI
        filename = path.name
        self.info_label.setText(f"✅ Выбран файл: {filename}")
        self.info_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.clear_button.setEnabled(True)
        
        # Emit signal
        self.file_selected.emit(file_path)
        
        logger.info(f"Файл выбран: {file_path}")
    
    def _on_clear_clicked(self) -> None:
        """Обработчик очистки выбора."""
        self._selected_file = None
        self.info_label.setText(
            "Перетащите файл сюда или нажмите кнопку выбора\n"
            f"Поддерживаемые форматы: {', '.join(self.SUPPORTED_FORMATS)}"
        )
        self.info_label.setStyleSheet("color: #666;")
        self.clear_button.setEnabled(False)
        self.status_label.setText("")
        
        logger.info("Выбор файла очищен")
    
    def _on_drag_enter(self, event) -> None:
        """Обработчик входа перетаскивания."""
        if event.mimeData().hasUrls():
            # Проверяем, что перетаскивается файл поддерживаемого формата
            urls = event.mimeData().urls()
            if urls:
                path = Path(urls[0].toLocalFile())
                if path.suffix.lower() in ['.pdf', '.docx', '.doc']:
                    event.acceptProposedAction()
                    self.drop_frame.setStyleSheet("""
                        QFrame {
                            border: 2px dashed #007bff;
                            border-radius: 10px;
                            background-color: #eef6ff;
                        }
                    """)
                    return
        event.ignore()
    
    def _on_drag_leave(self, event) -> None:
        """Обработчик выхода перетаскивания."""
        self.drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)
    
    def _on_drop(self, event) -> None:
        """Обработчик сброса файла."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self._handle_file_selection(file_path)
        
        self.drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)
    
    def set_processing(self, processing: bool) -> None:
        """Установка режима обработки.
        
        Args:
            processing: True если идёт обработка.
        """
        self.progress_bar.setVisible(processing)
        self.select_button.setEnabled(not processing)
        self.clear_button.setEnabled(not processing and self._selected_file is not None)
        
        if processing:
            self.status_label.setText("⏳ Обработка документа...")
            self.status_label.setStyleSheet("color: #007bff;")
        else:
            self.status_label.setText("✅ Документ обработан")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
    
    def get_selected_file(self) -> Optional[str]:
        """Получение пути к выбранному файлу.
        
        Returns:
            Путь к файлу или None.
        """
        return self._selected_file
    
    def clear(self) -> None:
        """Очистка виджета."""
        self._on_clear_clicked()
