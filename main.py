# main.py
"""
Точка входа приложения для диагностики болевых синдромов.
Включает функционал дообучения модели на основе обратной связи врача.
"""
import sys
from PyQt6.QtWidgets import QApplication
from database.connection import DatabaseManager
from ui.main_window import MainWindow
from controller import AppController

def main():
    # Инициализация приложения
    app = QApplication(sys.argv)
    app.setApplicationName("Clinical Diagnosis System")
    
    # Инициализация базы данных
    db_manager = DatabaseManager()
    db_manager.init_db()
    
    # Создание главного окна
    window = MainWindow()
    window.show()
    
    # Заглушка для model_engine (в реальном проекте здесь будет инициализация ML движка)
    class MockModelEngine:
        def extract_features(self, clinical_data):
            return {"dummy": 0}
        def load_model(self, path):
            pass
    
    model_engine = MockModelEngine()
    
    # Инициализация контроллера
    controller = AppController(window, model_engine, db_manager)
    
    # Обновление статистики обучения при запуске
    controller.refresh_training_stats()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
