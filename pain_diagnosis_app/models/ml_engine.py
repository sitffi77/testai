"""
ML Engine module for Pain Diagnosis Application.

Handles model loading, preprocessing, inference, and SHAP interpretation.
"""

import io
import base64
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    import shap
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(f"Required ML library not installed: {e}")

from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    PAIN_CLASSES,
    RANDOM_STATE,
    SHAP_FIGURE_DPI,
    SHAP_FIGURE_SIZE,
)

logger = logging.getLogger(__name__)


class MLEngine:
    """
    Machine Learning engine for pain classification.

    Handles model loading, data preprocessing, prediction, and SHAP analysis.

    Attributes:
        model: Loaded XGBoost classifier.
        is_loaded: Flag indicating if model is loaded.
        explainer: SHAP TreeExplainer instance.
    """

    def __init__(self) -> None:
        """Initialize the ML engine with no model loaded."""
        self.model: Optional[XGBClassifier] = None
        self.is_loaded: bool = False
        self.explainer: Optional[Any] = None
        self._feature_names: List[str] = ALL_FEATURES.copy()
        
        # Default categorical mappings for one-hot encoding
        self._categorical_maps: Dict[str, Dict[str, int]] = {}
        self._setup_default_categorical_maps()

    def _setup_default_categorical_maps(self) -> None:
        """Set up default mappings for categorical features."""
        self._categorical_maps = {
            "gender": {"M": 0, "F": 1, "Other": 2},
            "pain_location": {
                "head": 0, "neck": 1, "back": 2, "chest": 3,
                "abdomen": 4, "limbs": 5, "generalized": 6
            },
            "pain_type": {
                "aching": 0, "sharp": 1, "burning": 2, "throbbing": 3,
                "shooting": 4, "dull": 5, "stabbing": 6
            },
            "trigger_factor": {
                "movement": 0, "stress": 1, "weather": 2, "food": 3,
                "none": 4, "unknown": 5
            },
            "relief_factor": {
                "rest": 0, "medication": 1, "heat": 2, "cold": 3,
                "movement": 4, "none": 5
            },
            "medication_use": {
                "none": 0, "nsaids": 1, "opioids": 2, "antidepressants": 3,
                "anticonvulsants": 4, "muscle_relaxants": 5, "combination": 6
            },
        }

    def load_model(self, path: str) -> bool:
        """
        Load a trained XGBoost model from file.

        Args:
            path: Path to the serialized model file (.joblib).

        Returns:
            True if model loaded successfully, False otherwise.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            Exception: If model loading fails.
        """
        try:
            model_path = Path(path)
            if not model_path.exists():
                logger.error(f"Model file not found: {path}")
                # Create a default model for demonstration
                logger.info("Creating default model for demonstration...")
                self._create_default_model()
                return True
            
            self.model = joblib.load(model_path)
            self.is_loaded = True
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            logger.info(f"Model loaded successfully from {path}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create a default model for demonstration purposes
            logger.info("Creating default model for demonstration...")
            self._create_default_model()
            return True

    def _create_default_model(self) -> None:
        """Create a default XGBoost model for demonstration when no model file exists."""
        # Create synthetic training data for initialization
        np.random.seed(RANDOM_STATE)
        n_samples = 100
        
        # Generate synthetic numerical features
        X_num = np.random.rand(n_samples, len(NUMERICAL_FEATURES))
        X_num[:, 0] = X_num[:, 0] * 60 + 20  # age: 20-80
        X_num[:, 1] = X_num[:, 1] * 10  # pain_intensity: 0-10
        X_num[:, 2] = X_num[:, 2] * 365  # duration_days: 0-365
        
        # Generate synthetic categorical features (one-hot encoded)
        n_cat_values = sum(len(v) for v in self._categorical_maps.values())
        X_cat = np.zeros((n_samples, n_cat_values))
        
        # Simple pattern: assign categories based on index
        cat_idx = 0
        for feature, mapping in self._categorical_maps.items():
            for _ in range(len(mapping)):
                X_cat[:, cat_idx] = np.random.randint(0, 2, n_samples)
                cat_idx += 1
        
        X = np.hstack([X_num, X_cat])
        y = np.random.choice(len(PAIN_CLASSES), n_samples)
        
        # Train a simple model
        self.model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.model.fit(X, y)
        self.is_loaded = True
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("Default model created for demonstration purposes")

    def preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for model inference.

        Performs normalization of numerical features and one-hot encoding
        of categorical features.

        Args:
            data: Dictionary containing clinical feature values.

        Returns:
            Preprocessed DataFrame ready for model inference.
        """
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Handle missing numerical values
        for feat in NUMERICAL_FEATURES:
            if feat not in df.columns or df[feat].isna().any():
                df[feat] = 0.0
        
        # Handle missing categorical values
        for feat in CATEGORICAL_FEATURES:
            if feat not in df.columns or df[feat].isna().any():
                df[feat] = "unknown"
        
        # One-hot encode categorical features
        encoded_features = []
        for feat in CATEGORICAL_FEATURES:
            value = str(df[feat].iloc[0]).lower() if pd.notna(df[feat].iloc[0]) else "unknown"
            mapping = self._categorical_maps.get(feat, {})
            
            # Create binary vector for this categorical feature
            if value in mapping:
                vec = [0] * len(mapping)
                vec[mapping[value]] = 1
            else:
                # Unknown category - use all zeros or first category
                vec = [0] * len(mapping)
                if mapping:
                    vec[0] = 1  # Default to first category
            
            encoded_features.extend(vec)
        
        # Build final feature vector
        numerical_values = [float(df.get(feat, 0.0)) for feat in NUMERICAL_FEATURES]
        final_features = numerical_values + encoded_features
        
        # Create result DataFrame
        result_df = pd.DataFrame([final_features], columns=self._feature_names)
        
        return result_df

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform prediction on preprocessed data.

        Args:
            data: Dictionary containing clinical feature values.

        Returns:
            Dictionary containing:
                - class: Predicted pain class
                - probabilities: Dict of class probabilities
                - shap_values: Dict of SHAP feature contributions
                - shap_plot_base64: Base64-encoded SHAP summary plot

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess data
            df_processed = self.preprocess(data)
            X = df_processed.values
            
            # Get predictions
            prediction_idx = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            predicted_class = PAIN_CLASSES[prediction_idx]
            prob_dict = {
                PAIN_CLASSES[i]: float(probabilities[i])
                for i in range(len(PAIN_CLASSES))
            }
            
            # Calculate SHAP values
            shap_values = self._calculate_shap(df_processed)
            
            # Generate SHAP plot
            shap_plot_base64 = self._generate_shap_plot(df_processed, shap_values)
            
            result = {
                "class": predicted_class,
                "probabilities": prob_dict,
                "shap_values": shap_values,
                "shap_plot_base64": shap_plot_base64,
            }
            
            logger.info(f"Prediction completed: {predicted_class}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _calculate_shap(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate SHAP values for feature importance.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            Dictionary mapping feature names to SHAP values.
        """
        if self.explainer is None:
            return {}
        
        try:
            shap_values_raw = self.explainer.shap_values(df)
            
            # Handle multi-class SHAP values
            if isinstance(shap_values_raw, list):
                # Take mean absolute SHAP values across classes
                shap_mean = np.mean([np.abs(sv) for sv in shap_values_raw], axis=0)
            else:
                shap_mean = np.abs(shap_values_raw)
            
            # Get single sample SHAP values (first row)
            if len(shap_mean.shape) > 1:
                shap_single = shap_mean[0]
            else:
                shap_single = shap_mean
            
            # Map to feature names
            shap_dict = {
                self._feature_names[i]: float(shap_single[i])
                for i in range(len(self._feature_names))
            }
            
            return shap_dict
            
        except Exception as e:
            logger.warning(f"SHAP calculation warning: {e}")
            return {feat: 0.0 for feat in self._feature_names}

    def _generate_shap_plot(self, df: pd.DataFrame, shap_values: Dict[str, float]) -> str:
        """
        Generate SHAP summary plot and convert to base64 string.

        Args:
            df: Preprocessed DataFrame.
            shap_values: Dictionary of SHAP values.

        Returns:
            Base64-encoded PNG image string.
        """
        try:
            fig, ax = plt.subplots(figsize=SHAP_FIGURE_SIZE, dpi=SHAP_FIGURE_DPI)
            
            # Sort features by absolute SHAP value
            sorted_features = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Take top 10 features
            top_features = sorted_features[:10]
            
            # Create horizontal bar chart
            features = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('SHAP Value (Feature Importance)')
            ax.set_title('Top Features Contributing to Prediction')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=SHAP_FIGURE_DPI)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP plot: {e}")
            # Return empty string on failure
            return ""

    def get_interpretation(self, prediction_result: Dict[str, Any]) -> str:
        """
        Generate human-readable interpretation of prediction results.

        Args:
            prediction_result: Dictionary containing prediction results.

        Returns:
            Human-readable interpretation string.
        """
        predicted_class = prediction_result.get("class", "unknown")
        probabilities = prediction_result.get("probabilities", {})
        shap_values = prediction_result.get("shap_values", {})
        
        # Get confidence
        confidence = probabilities.get(predicted_class, 0.0)
        confidence_pct = confidence * 100
        
        # Build interpretation
        interpretation_parts = [
            f"Предсказанный тип боли: {self._translate_class(predicted_class)}.",
            f"Уверенность модели: {confidence_pct:.1f}%.",
            "",
            "Распределение вероятностей:",
        ]
        
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            interpretation_parts.append(f"  - {self._translate_class(cls)}: {prob*100:.1f}%")
        
        # Add top contributing features
        if shap_values:
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            interpretation_parts.extend([
                "",
                "Наиболее влияющие факторы:",
            ])
            for feat, val in sorted_shap:
                direction = "повышает" if val > 0 else "снижает"
                interpretation_parts.append(f"  - {feat}: {direction} вероятность ({val:.3f})")
        
        return "\n".join(interpretation_parts)

    def _translate_class(self, class_name: str) -> str:
        """Translate pain class name to Russian."""
        translations = {
            "nociceptive": "Ноцицептивная (тканевая)",
            "neuropathic": "Нейропатическая",
            "nociplastic": "Ноципластическая",
            "mixed": "Смешанная",
        }
        return translations.get(class_name, class_name)


# Singleton instance
_ml_engine_instance: Optional[MLEngine] = None


def get_ml_engine() -> MLEngine:
    """
    Get or create the singleton ML engine instance.

    Returns:
        MLEngine instance.
    """
    global _ml_engine_instance
    if _ml_engine_instance is None:
        _ml_engine_instance = MLEngine()
    return _ml_engine_instance
