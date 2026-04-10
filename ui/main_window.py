# ui/main_window.py (фрагмент)
from PyQt6.QtWidgets import QMainWindow, QTabWidget
from ui.training_view import TrainingView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... существующий код ...
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Инициализация вкладок
        # self.input_form = InputForm()  # Предполагается, что эти классы уже существуют
        # self.results_view = ResultsView()
        # self.doc_upload = DocumentUpload()
        # self.flags_dashboard = FlagsDashboard()
        self.training_tab = TrainingView() # НОВАЯ
        
        # Добавление вкладок
        # self.tabs.addTab(self.input_form, "Диагностика")
        # self.tabs.addTab(self.doc_upload, "Документы")
        # self.tabs.addTab(self.flags_dashboard, "Флаги")
        self.tabs.addTab(self.training_tab, "Обучение модели") # НОВАЯ
