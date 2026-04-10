# controller.py (фрагмент дополнений)
import logging
from PyQt6.QtWidgets import QMessageBox
from ui.training_view import TrainingView
from worker import TrainingWorker
from models.db_models import Diagnosis, TrainingData

class AppController:
    def __init__(self, view, model_engine, db_manager):
        self.view = view
        self.model_engine = model_engine
        self.db_manager = db_manager
        self.training_worker = None
        
        self._connect_training_signals()
        
    def _connect_training_signals(self):
        # Подключение сигналов вкладки обучения
        train_view = self.view.training_tab # Предполагаем, что вкладка добавлена в main_window
        train_view.start_training_signal.connect(self.start_training_process)
        
    def submit_diagnosis_feedback(self, diagnosis_id: int, is_correct: bool, true_label: str = None):
        """
        Сохраняет обратную связь врача для будущего обучения.
        Если диагноз неверен, врач указывает правильный (true_label).
        """
        session = self.db_manager.get_session()
        try:
            # Получаем исходные данные диагноза
            diagnosis_obj = session.query(Diagnosis).get(diagnosis_id)
            if not diagnosis_obj:
                return False
                
            clinical_data = diagnosis_obj.clinical_data
            patient = clinical_data.patient
            
            # Формируем вектор признаков (нужно повторить логику предобработки ML Engine)
            # В идеале ML Engine должен иметь метод get_features_dict(clinical_data)
            features = self.model_engine.extract_features(clinical_data) 
            
            target = true_label if true_label else diagnosis_obj.predicted_class
            
            new_sample = TrainingData(
                patient_id=patient.id,
                clinical_data_id=clinical_data.id,
                features_json=features,
                true_label=target,
                source='correction' if not is_correct else 'verification'
            )
            
            session.add(new_sample)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error saving feedback: {e}")
            return False
        finally:
            session.close()

    def start_training_process(self):
        """Запускает асинхронное обучение"""
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.warning(self.view, "Внимание", "Обучение уже идет.")
            return
            
        self.training_worker = TrainingWorker(model_path="models/pain_model_v1.joblib")
        self.training_worker.log_update.connect(self.view.training_tab.log_message)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.start()
        
    def on_training_finished(self, success: bool, message: str):
        self.view.training_tab.training_finished(success, message)
        if success:
            # Перезагружаем модель в движке
            self.model_engine.load_model("models/pain_model_v1.joblib")
            # Обновляем статистику в UI
            self.refresh_training_stats()
            
    def refresh_training_stats(self):
        # Запрос к БД для получения кол-ва примеров
        session = self.db_manager.get_session()
        count = session.query(TrainingData).count()
        # ... логика получения классов и последней точности ...
        self.view.training_tab.update_stats(count, 0, 0.0) # Заглушка
        session.close()
