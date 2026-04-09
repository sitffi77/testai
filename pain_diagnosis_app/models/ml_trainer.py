"""
ML Trainer module for Pain Diagnosis Application.

Handles model fine-tuning using labeled data collected from doctor feedback.
"""

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

from models.db_models import TrainingData
from config import BASE_DIR, PAIN_CLASSES

logger = logging.getLogger(__name__)


class MLTrainer:
    """
    Class for managing the model fine-tuning process.
    
    Collects labeled data from the database, retrains the model,
    and saves the updated version with backup of the old one.
    
    Attributes:
        config: Training configuration dictionary.
        current_model: Currently loaded model (optional).
        preprocessor: Preprocessor instance (optional).
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Initialize the ML trainer.
        
        Args:
            config_path: Path to the training configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.current_model = None
        self.preprocessor = None
        self._label_encoder = {label: idx for idx, label in enumerate(PAIN_CLASSES)}
        self._idx_to_label = {idx: label for label, idx in self._label_encoder.items()}
        
    def _load_config(self, path: str) -> Dict:
        """
        Load training configuration from YAML file.
        
        Args:
            path: Path to the configuration file.
            
        Returns:
            Configuration dictionary.
        """
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = BASE_DIR / path
            
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)['training']
        except FileNotFoundError:
            logger.warning(f"Config {config_path} not found. Using defaults.")
            return {
                "min_samples": 20,
                "validation_split": 0.2,
                "strategy": "retrain",
                "params": {
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss"
                },
                "min_accuracy_threshold": 0.75,
                "backup_dir": "models/backups"
            }
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def fetch_training_data(self, session: Session) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract labeled training data from the database.
        
        Args:
            session: SQLAlchemy database session.
            
        Returns:
            Tuple of (features DataFrame, labels Series).
            
        Raises:
            ValueError: If not enough samples are available.
        """
        logger.info("Fetching training data from DB...")
        
        samples = session.query(TrainingData).filter(
            TrainingData.true_label.isnot(None)
        ).all()
        
        if len(samples) < self.config['min_samples']:
            raise ValueError(
                f"Not enough samples: {len(samples)}. "
                f"Required minimum: {self.config['min_samples']}"
            )
        
        features_list = []
        labels_list = []
        
        for sample in samples:
            features = json.loads(sample.features_json)
            features_list.append(features)
            labels_list.append(sample.true_label)
            
        df = pd.DataFrame(features_list)
        labels = pd.Series(labels_list)
        
        logger.info(f"Fetched {len(df)} samples with {len(df.columns)} features.")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
        
        return df, labels

    def _encode_labels(self, y: pd.Series) -> np.ndarray:
        """
        Encode string labels to numeric indices.
        
        Args:
            y: Series of string labels.
            
        Returns:
            Array of numeric labels.
        """
        # Build encoder from data if needed
        unique_labels = sorted(list(set(y)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Update instance encoders
        self._label_encoder = label_to_idx
        self._idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return np.array([label_to_idx[label] for label in y])

    def _decode_labels(self, y_encoded: np.ndarray) -> List[str]:
        """
        Decode numeric labels back to strings.
        
        Args:
            y_encoded: Array of numeric labels.
            
        Returns:
            List of string labels.
        """
        return [self._idx_to_label.get(idx, f"class_{idx}") for idx in y_encoded]

    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """
        Main training method.
        
        Args:
            X: Features DataFrame.
            y: Labels Series.
            
        Returns:
            Tuple of (trained model, metrics dictionary).
        """
        logger.info("Starting training process...")
        
        # Encode labels
        y_encoded = self._encode_labels(y)
        num_classes = len(np.unique(y_encoded))
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, 
            test_size=self.config['validation_split'], 
            random_state=42, 
            stratify=y_encoded
        )
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = self.config['params'].copy()
        params['num_class'] = num_classes
        
        logger.info(
            f"Training with {len(X_train)} samples, {len(X_val)} validation. "
            f"Classes: {num_classes}"
        )
        
        # Training with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Predictions and metrics
        preds_encoded = model.predict(dval)
        if num_classes > 1:
            y_pred_encoded = np.argmax(preds_encoded, axis=1)
        else:
            y_pred_encoded = preds_encoded.round().astype(int)
        
        # Decode predictions
        y_pred = self._decode_labels(y_pred_encoded)
        y_val_str = self._decode_labels(y_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_str, y_pred)
        
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        
        if accuracy < self.config['min_accuracy_threshold']:
            logger.warning(
                f"Accuracy {accuracy:.4f} is below threshold "
                f"{self.config['min_accuracy_threshold']}. "
                f"Model saved but flagged."
            )
        
        # Classification report
        try:
            report = classification_report(y_val_str, y_pred, output_dict=True)
            logger.info(f"Classification report: {report}")
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            report = {}
        
        metrics = {
            "accuracy": float(accuracy),
            "samples_count": len(X),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "classes": num_classes,
            "classification_report": report,
            "meets_threshold": accuracy >= self.config['min_accuracy_threshold']
        }
        
        return model, metrics

    def run_full_pipeline(
        self, 
        session: Session, 
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Full training pipeline: Fetch -> Train -> Save.
        
        Args:
            session: SQLAlchemy database session.
            output_path: Path to save the trained model.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Metrics dictionary.
        """
        if progress_callback:
            progress_callback("Сбор данных из базы...")
        
        # 1. Fetch data
        X, y = self.fetch_training_data(session)
        
        if progress_callback:
            progress_callback(f"Найдено {len(X)} примеров для обучения...")
        
        # 2. Train
        if progress_callback:
            progress_callback("Обучение модели...")
        
        model, metrics = self.train(X, y)
        
        # 3. Backup old model
        if os.path.exists(output_path):
            backup_dir = Path(self.config.get('backup_dir', 'models/backups'))
            if not backup_dir.is_absolute():
                backup_dir = BASE_DIR / backup_dir
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"pain_classifier.backup_{timestamp}.joblib"
            shutil.copy(output_path, backup_path)
            logger.info(f"Old model backed up to {backup_path}")
            
            if progress_callback:
                progress_callback(f"Старая модель сохранена в бэкап: {backup_path.name}")
        
        # 4. Save new model
        if progress_callback:
            progress_callback("Сохранение новой модели...")
        
        joblib.dump(model, output_path)
        logger.info(f"New model saved to {output_path}")
        
        metrics["model_path"] = output_path
        
        if progress_callback:
            progress_callback("Обучение завершено успешно!")
        
        return metrics
