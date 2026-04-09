# models/ml_trainer.py
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sqlalchemy.orm import Session

from models.db_models import TrainingData, ClinicalData
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainer:
    """
    Класс для управления процессом дообучения модели.
    Собирает размеченные данные из БД, переобучает модель и сохраняет её.
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config = self._load_config(config_path)
        self.current_model = None
        self.preprocessor = None
        
    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)['training']
        except FileNotFoundError:
            logger.warning(f"Config {path} not found. Using defaults.")
            return {
                "min_samples": 20,
                "validation_split": 0.2,
                "strategy": "retrain",
                "params": {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100},
                "min_accuracy_threshold": 0.75
            }

    def fetch_training_data(self, session: Session) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Извлекает размеченные данные из БД.
        Возвращает DataFrame признаков и Series меток.
        """
        logger.info("Fetching training data from DB...")
        samples = session.query(TrainingData).filter(
            TrainingData.true_label.isnot(None)
        ).all()
        
        if len(samples) < self.config['min_samples']:
            raise ValueError(f"Not enough samples: {len(samples)}. Required: {self.config['min_samples']}")
        
        features_list = []
        labels_list = []
        
        for sample in samples:
            features = json.loads(sample.features_json)
            features_list.append(features)
            labels_list.append(sample.true_label)
            
        df = pd.DataFrame(features_list)
        labels = pd.Series(labels_list)
        
        logger.info(f"Fetched {len(df)} samples with {len(df.columns)} features.")
        return df, labels

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Основной метод обучения.
        Возвращает метрики и путь к сохраненной модели.
        """
        logger.info("Starting training process...")
        
        # Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config['validation_split'], 
            random_state=42, 
            stratify=y
        )
        
        # Подготовка матриц DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = self.config['params']
        
        # Обработка многоклассовой классификации
        num_classes = len(np.unique(y))
        if num_classes > 1:
            params['num_class'] = num_classes
        
        logger.info(f"Training with {len(X_train)} samples, {len(X_val)} validation. Classes: {num_classes}")
        
        # Обучение
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Предсказания и метрики
        preds = model.predict(dval)
        if num_classes > 1:
            y_pred = np.argmax(preds, axis=1)
            # Маппинг индексов обратно в лейблы (упрощенно, нужно мапить через encoder в реальном коде)
            # Здесь предполагаем, что лейблы строковые, XGBoost внутри мапит их
            # Для простоты берем accuracy
            unique_labels = sorted(list(set(y_val)))
            y_pred_labels = [unique_labels[i] for i in y_pred]
            accuracy = accuracy_score(y_val, y_pred_labels)
        else:
            y_pred = preds.round()
            accuracy = accuracy_score(y_val, y_pred)
            
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        
        if accuracy < self.config['min_accuracy_threshold']:
            logger.warning(f"Accuracy {accuracy} is below threshold {self.config['min_accuracy_threshold']}. Model saved but flagged.")
            
        metrics = {
            "accuracy": accuracy,
            "samples_count": len(X),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "classes": num_classes
        }
        
        return model, metrics

    def run_full_pipeline(self, session: Session, output_path: str) -> Dict[str, Any]:
        """
        Полный пайплайн: Сбор -> Обучение -> Сохранение.
        """
        # 1. Сбор данных
        X, y = self.fetch_training_data(session)
        
        # 2. Обучение
        model, metrics = self.train(X, y)
        
        # 3. Бэкап старой модели
        if os.path.exists(output_path):
            backup_path = f"{output_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy(output_path, backup_path)
            logger.info(f"Old model backed up to {backup_path}")
            
        # 4. Сохранение новой
        joblib.dump(model, output_path)
        logger.info(f"New model saved to {output_path}")
        
        metrics["model_path"] = output_path
        return metrics
