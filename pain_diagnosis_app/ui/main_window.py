"""
Main window for Pain Diagnosis Application.

Provides the main application window with tabs, menu, and status bar.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QMenuBar,
    QMenu,
    QStatusBar,
    QLabel,
    QMessageBox,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont

from config import WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT
from ui.input_form import ClinicalInputForm
from ui.results_view import ResultsView

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window.

    Contains:
        - Menu bar with File, Help menus
        - Tab widget with Input and Results tabs
        - Status bar for messages

    Signals:
        diagnose_requested: Emitted when user requests diagnosis from input form.
        save_requested: Emitted when user requests to save results.
    """

    diagnose_requested = pyqtSignal(dict)
    save_requested = pyqtSignal()

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        
        self.input_form: Optional[ClinicalInputForm] = None
        self.results_view: Optional[ResultsView] = None
        
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        
        logger.info("Main window initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Window properties
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.input_form = ClinicalInputForm()
        self.results_view = ResultsView()

        self.tabs.addTab(self.input_form, "📋 Ввод данных")
        self.tabs.addTab(self.results_view, "📊 Результаты")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status label
        self.status_label = QLabel("Готов к работе")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.status_bar.addWidget(self.status_label, 1)

        # Progress indicator
        self.progress_label = QLabel("")
        self.status_bar.addPermanentWidget(self.progress_label)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        menubar.setFont(QFont("Arial", 10))

        # File menu
        file_menu = menubar.addMenu("Файл")

        # New action
        new_action = QAction("Новая диагностика", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_diagnosis)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("Вид")

        # Go to input tab
        input_tab_action = QAction("Перейти к вводу данных", self)
        input_tab_action.setShortcut("Ctrl+1")
        input_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(input_tab_action)

        # Go to results tab
        results_tab_action = QAction("Перейти к результатам", self)
        results_tab_action.setShortcut("Ctrl+2")
        results_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(results_tab_action)

        # Help menu
        help_menu = menubar.addMenu("Справка")

        # About action
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # Help action
        help_action = QAction("Помощь", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        if self.input_form:
            self.input_form.submit_clicked.connect(self._on_form_submit)
            self.input_form.reset_clicked.connect(self._on_form_reset)

        if self.results_view:
            self.results_view.save_requested.connect(self._on_save_requested)

    def _on_form_submit(self, data: dict) -> None:
        """
        Handle form submission.

        Args:
            data: Dictionary containing clinical data.
        """
        logger.info(f"Form submitted with data: {list(data.keys())}")
        self.diagnose_requested.emit(data)

    def _on_form_reset(self) -> None:
        """Handle form reset."""
        logger.info("Form reset")
        if self.results_view:
            self.results_view.clear_results()
        self.set_status_message("Форма сброшена")

    def _on_save_requested(self) -> None:
        """Handle save request from results view."""
        logger.info("Save requested")
        self.save_requested.emit()

    def _on_new_diagnosis(self) -> None:
        """Handle new diagnosis action."""
        reply = QMessageBox.question(
            self,
            "Новая диагностика",
            "Начать новую диагностику? Текущие данные будут сброшены.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.input_form:
                self.input_form.reset_form()
            if self.results_view:
                self.results_view.clear_results()
            self.tabs.setCurrentIndex(0)
            self.set_status_message("Готов к новой диагностике")

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """
        <h2>Система поддержки диагностирования болевых синдромов</h2>
        <p>Версия: 1.0.0</p>
        <p>Приложение для классификации типов боли с использованием ML.</p>
        <p><b>Технологии:</b></p>
        <ul>
            <li>Python 3.10+</li>
            <li>PyQt6 (UI)</li>
            <li>XGBoost (ML)</li>
            <li>SHAP (интерпретация)</li>
            <li>SQLAlchemy (БД)</li>
        </ul>
        <p>© 2024 Medical AI Lab</p>
        """
        
        QMessageBox.about(self, "О программе", about_text)

    def _show_help(self) -> None:
        """Show help dialog."""
        help_text = """
        <h2>Как использовать приложение</h2>
        
        <h3>1. Ввод данных</h3>
        <p>Заполните форму клиническими данными пациента:</p>
        <ul>
            <li>ФИО и демографические данные</li>
            <li>Числовые показатели (интенсивность боли, длительность и т.д.)</li>
            <li>Категориальные признаки (локализация, тип боли и т.д.)</li>
        </ul>
        
        <h3>2. Диагностика</h3>
        <p>Нажмите кнопку "Диагностировать" для запуска ML-классификации.</p>
        
        <h3>3. Результаты</h3>
        <p>Просмотрите результаты:</p>
        <ul>
            <li>Предсказанный тип боли</li>
            <li>Вероятности для каждого класса</li>
            <li>График важности признаков (SHAP)</li>
            <li>Текстовая интерпретация</li>
        </ul>
        
        <h3>4. Сохранение</h3>
        <p>Нажмите "Сохранить в базу данных" для сохранения результатов.</p>
        
        <p><b>Горячие клавиши:</b></p>
        <ul>
            <li>Ctrl+N - Новая диагностика</li>
            <li>Ctrl+1 - Перейти к вводу</li>
            <li>Ctrl+2 - Перейти к результатам</li>
            <li>Ctrl+Q - Выход</li>
        </ul>
        """
        
        QMessageBox.about(self, "Помощь", help_text)

    def set_status_message(self, message: str) -> None:
        """
        Set status bar message.

        Args:
            message: Message to display.
        """
        self.status_label.setText(message)
        logger.debug(f"Status: {message}")

    def display_results(self, result: dict) -> None:
        """
        Display prediction results.

        Args:
            result: Prediction result dictionary.
        """
        if self.results_view:
            self.results_view.display_results(result)
            self.tabs.setCurrentIndex(1)

    def set_loading_state(self, loading: bool) -> None:
        """
        Set application loading state.

        Args:
            loading: True if loading, False otherwise.
        """
        if loading:
            self.progress_label.setText("⏳ Обработка...")
            if self.input_form:
                self.input_form.set_enabled(False)
            if self.results_view:
                self.results_view.set_loading_state(True)
        else:
            self.progress_label.setText("")
            if self.input_form:
                self.input_form.set_enabled(True)

    def show_error(self, title: str, message: str) -> None:
        """
        Show error dialog.

        Args:
            title: Dialog title.
            message: Error message.
        """
        QMessageBox.critical(self, title, message)
        logger.error(f"{title}: {message}")

    def show_warning(self, title: str, message: str) -> None:
        """
        Show warning dialog.

        Args:
            title: Dialog title.
            message: Warning message.
        """
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event) -> None:
        """
        Handle window close event.

        Args:
            event: Close event.
        """
        reply = QMessageBox.question(
            self,
            "Выход",
            "Вы уверены, что хотите выйти из приложения?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Application closing")
            event.accept()
        else:
            event.ignore()
