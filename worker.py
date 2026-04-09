# worker.py
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from models.ml_trainer import MLTrainer
from database.connection import DatabaseManager
import traceback

class TrainingWorker(QThread):
    finished = pyqtSignal(bool, str) # success, message
    log_update = pyqtSignal(str)
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.trainer = MLTrainer()
        
    @pyqtSlot()
    def run(self):
        try:
            self.log_update.emit("Подключение к базе данных...")
            db_manager = DatabaseManager()
            session = db_manager.get_session()
            
            self.log_update.emit("Сбор и подготовка данных...")
            # В реальном приложении пути к конфигам лучше брать из глобального конфига
            metrics = self.trainer.run_full_pipeline(session, self.model_path)
            
            session.close()
            
            msg = (f"Обучение завершено.\n"
                   f"Примеров: {metrics['samples_count']}\n"
                   f"Точность: {metrics['accuracy']:.2%}\n"
                   f"Модель сохранена: {metrics['model_path']}")
            
            self.finished.emit(True, msg)
            
        except Exception as e:
            error_msg = f"Ошибка обучения: {str(e)}\n{traceback.format_exc()}"
            self.log_update.emit(error_msg)
            self.finished.emit(False, str(e))
